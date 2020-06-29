"""
Neural Network Training framework
Author: Jitender Singh Virk [virksaab.github.io]
Last update: 29 Jun 2020
"""


import time, copy, torch, gc
import pandas as pd
import numpy as np
try: from apex import amp # Mixed precision training https://github.com/NVIDIA/apex
except: print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
from torch.optim.lr_scheduler import OneCycleLR
from typing import Tuple, Callable
from fastprogress import master_bar, progress_bar

#from yamlf
import .metrics as yamlf_metrics
from .utils import calc_time_taken
from .vision import LoadData


######################## FUNCTIONS #################################
def flatten_check(y_pred, y_true):
    "Check that `y_pred` and `y_true` have the same number of elements and flatten them."
    y_pred, y_true = y_pred.contiguous().view(-1), y_true.contiguous().view(-1)
    assert len(y_pred) == len(y_true), f"Expected output and target to have the same number of elements but got {len(y_pred)} and {len(y_true)}."
    return y_pred, y_true

def update_dict(original_dict, new_dict):
    """update metrics's history dict"""
    if new_dict == None: return original_dict
    for k in new_dict:
        if k not in original_dict:
            original_dict[k] = []
            original_dict[k].append(new_dict[k])
        else: original_dict[k].append(new_dict[k])
    return original_dict


########################### CLASSES ##############################
# Base trainer class
class Trainer(object):
    def __init__(self,
        data:Tuple[dict, LoadData],
        net:torch.nn.Module,
        criterion:torch.nn.Module,
        opt_func:torch.optim.Optimizer=torch.optim.Adam,
        metrics:Tuple[yamlf_metrics.SKMetrics, yamlf_metrics.Metrics, str, Callable, list]=None,
        defs:dict=None,
        mixed_precision:bool=False,
        batch_lrs_func:torch.optim.lr_scheduler._LRScheduler=OneCycleLR,
        epoch_lrs_func:torch.optim.lr_scheduler._LRScheduler=None,
        tblogtype:str="batch"
    ):

        """
        Trainer class for basic training and validation
        Args:
            data (dict or EasyDict or LoadData): dataloaders and data
            net (nn.Module): network or model to train
            loss_func (Callable nn.Module): criterion or loss function
            opt_func (torch.optim.Optimizer): optimizer
            metrics (dict): evaluation functions like accuracy
            defs (EasyDict): default settings dict
            batch_lrs_func (torch): batch learning rate scheduler function
            epoch_lrs_func (torch lr_scheduler): epoch learning rate scheduler function
            tblogtype (str): update Tensorboard logs per `batch` or `epoch`. Default is batch
        """
        # Set dataloaders
        if isinstance(data, dict):
            self.train_dl = data['train']
            if 'val' in data.keys(): self.val_dl = data['val']
            if 'test' in data.keys(): self.test_dl = data['test']
            if defs == None: raise TypeError('Please pass default settings dict. defs=<DefaultSettings>')
            self.__dict__.update(defs)
        elif isinstance(data, LoadData): self.__dict__.update(data.__dict__)
        else: raise NotImplementedError('unsupported data!')
        # set network
        self.net = net.to(self.device)
        # set loss or criterion
        self.criterion = criterion
        if opt_func.__name__ in ['SGD', 'RMSprop']:
            self.optimizer = opt_func(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr, momentum=max(self.moms))
        else:
            self.optimizer = opt_func(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr, betas=self.moms)
        # set metrics
        self.metrics = metrics
        if isinstance(self.metrics, (str, list)):
            if self.metrics in yamlf_metrics.SKMetrics.__dict__.keys():
                self.metrics = yamlf_metrics.SKMetrics([self.metrics])
            elif self.metrics in yamlf_metrics.Metrics.__dict__.keys():
                self.metrics = yamlf_metrics.Metrics([self.metrics])
            else:
                raise AttributeError(f"{self.metrics} is unknown metric.")

        # Set LR scheduler
        self.batch_lrs_func = batch_lrs_func
        self.epoch_lrs_func = epoch_lrs_func
        if self.batch_lrs_func != None and self.epoch_lrs_func != None:
            print("[WARNING]: Both batch and epoch scheduler are given. This might cause inconsistent training.")
        # Other settings
        self.low_storage = self.low_storage # If low storage, don't save chkpts automatically
        self.tblogtype = tblogtype # tensorboard log type (batch or epoch)
        if not self.low_storage and self.tblogtype == 'batch':
            self.tbc = 0 # Traning batch counter for tensorboard logs
            self.vbc = 0 # Validation batch counter for tensorboard logs
        # Mixed Precision
        self.mixed_precision = mixed_precision
        if mixed_precision:
            self.net, self.optimizer = amp.initialize(self.net, self.optimizer)

    def one_batch(self, databatch):
        if isinstance(databatch, list):
            X, y_true = databatch
            X, y_true = X.to(self.device), y_true.to(self.device)
        else:
            X = databatch[self.DN].to(self.device)
            y_true = databatch[self.LN].to(self.device)

        y_pred = self.net(X)
        loss = self.criterion(y_pred, y_true)

        if self.net.training:
            if self.mixed_precision:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else: loss.backward()
            self.optimizer.step()

        # return preds and true labels for evaluation
        if self.metrics == None: return loss.item(), None
        else: return loss.item(), {'y_pred':y_pred, 'y_true': y_true}

    def train_one_epoch(self, train_iters):
        oe_loss, oe_metrics = [], {}
        self.net.train() # train mode
        enable_batch_tblogs = (not self.low_storage) and (self.tblogtype == 'batch') # just to save two condition check in loop
        for b, databatch in progress_bar(enumerate(self.train_dl), total=len(self.train_dl), parent=self.mbar):
            self.optimizer.zero_grad()
            b_loss, b_ys = self.one_batch(databatch)
            # log loss and metrics
            oe_loss.append(b_loss)
            b_metrics = self.compute_metrics(b_ys)
            oe_metrics = update_dict(oe_metrics, b_metrics)
            self.mbar.child.comment = f'| loss: {b_loss:.5f}'
            if train_iters and b == (train_iters - 1): break
            if hasattr(self, 'batch_lrs'): self.batch_lrs.step()
            if enable_batch_tblogs:
                self.tbwriter.add_scalar('LearningRate/Batch', self.optimizer.param_groups[0]['lr'], self.tbc)
                self.tbwriter.add_scalar('Loss/train', b_loss, self.tbc)
                if self.metrics != None: self.tbwriter.add_scalars('Metrics/train',   b_metrics, self.tbc)
                self.tbc += 1

        return np.mean(oe_loss), {k:np.mean(oe_metrics[k]) for k in oe_metrics}

    def validate(self, val_iters):
        self.net.eval() # eval mode
        val_loss, val_metrics = [], []
        enable_batch_tblogs = (not self.low_storage) and (self.tblogtype == 'batch') # just to save two condition check in loop
        if not isinstance(self.val_dl, list): self.val_dl = [self.val_dl]
        for dl in self.val_dl:
            _val_loss, _val_metrics = [], {}
            for b, databatch in enumerate(progress_bar(dl, total=len(dl), parent=self.mbar)):
                b_loss, b_ys = self.one_batch(databatch)
                _val_loss.append(b_loss)
                b_metrics = self.compute_metrics(b_ys)
                _val_metrics = update_dict(_val_metrics, b_metrics)
                self.mbar.child.comment = f'| loss: {b_loss:.5f}'
                if enable_batch_tblogs:
                    self.tbwriter.add_scalar('Loss/val', b_loss, self.vbc)
                    if self.metrics != None: self.tbwriter.add_scalars('Metrics/val',   b_metrics, self.vbc)
                    self.vbc += 1
                if val_iters and b == (val_iters - 1): break
            val_loss.append(np.mean(_val_loss))
            val_metrics.append({k:np.mean(_val_metrics[k]) for k in _val_metrics})
        if len(self.val_dl) == 1: return val_loss[0], val_metrics[0]
        return val_loss, val_metrics

    def fit(self, epochs=None, lr=None, moms=None, wd=None, init_epoch=0, train_iters=None, val_iters=None):
        # set params
        if epochs: self.epochs = epochs
        if lr: self.lr = lr
        if moms: self.moms = moms
        if wd: self.wd = wd
        if init_epoch > 0: self.init_epoch = init_epoch
        # set optimizer params
        for pg in self.optimizer.param_groups:
            pg['lr'] = self.lr
            if 'momentum' in pg.keys(): pg['momentum'] = max(self.moms)
            else: pg['betas'] = self.moms
            pg['weight_decay'] = self.wd
        # LR scheduler
        if self.batch_lrs_func:
            self.batch_lrs = self.batch_lrs_func(self.optimizer, max_lr=lr, steps_per_epoch=len(self.train_dl), epochs=epochs)
        if self.epoch_lrs_func: self.epoch_lrs = self.epoch_lrs_func(self.optimizer, 'min', verbose=True)

        # start training
        self.mbar = master_bar(range(self.init_epoch, epochs)) # Progress bar
        first_epoch_flag = True # just to set names for the first time
        best_val_loss = float('Inf')
        enable_epoch_tblogs = (not self.low_storage) and (self.tblogtype == 'epoch') # just to save two condition check in loop
        for epoch in self.mbar:
            _time_taken = time.time() # start count time taken to run
            toe_loss, toe_metrics = self.train_one_epoch(train_iters)
            val_loss, val_metrics = self.validate(val_iters)
            # tensorboard writer
            if enable_epoch_tblogs:
                self.tbwriter.add_scalar('LearningRate/Epoch', self.optimizer.param_groups[0]['lr'], epoch)
                self.tbwriter.add_scalar('Loss/train', toe_loss, epoch)
                self.tbwriter.add_scalar('Loss/val',   val_loss, epoch)
                if self.metrics != None:
                    self.tbwriter.add_scalars('Metrics/train', toe_metrics, epoch)
                    self.tbwriter.add_scalars('Metrics/val',   val_metrics, epoch)

            if isinstance(val_loss, list): val_loss = val_loss[0] # temporary fix for single valset
            if val_loss < best_val_loss:
                if not self.low_storage: self.save_weights(self.chkptdir/'best_val_loss.pt')
                best_val_loss = val_loss

            if hasattr(self, 'epoch_lrs'): self.epoch_lrs.step(val_loss)
            if first_epoch_flag:
                # set the header (column names) of output dataframe
                header = ['epoch', 'train_loss']
                header += [f"train_{k}" for k in toe_metrics.keys()]
                if len(self.val_dl) > 1:
                    header += [f'val_loss{i}' for i in range(len(self.val_dl))]
                    header += [f"val_{k}" for vm in val_metrics for k in vm.keys()]
                else:
                    header += ['val_loss']
                    header += [f"val_{k}" for k in val_metrics.keys()]
                header += ['time'] # display time taken for one epoch
                self.mbar.write(pd.DataFrame(columns=header), table=True)
                first_epoch_flag = False
            # count time taken by one epoch
            _txt = [epoch, round(toe_loss, 5)]+[round(_t, 5) for _t in toe_metrics.values()]
            if len(self.val_dl) > 1:
                _txt += [round(vl, 5) for vl in val_loss]
                _txt += [round(v, 5) for vm in val_metrics for v in vm.values()]
            else:
                _txt += [round(val_loss, 5)]
                _txt += [round(v, 5) for v in val_metrics.values()]
            _txt.append(calc_time_taken(_time_taken))
            self.mbar.write([str(v) for v in _txt], table=True) # time taken for one epoch
            # create checkpoint
            if not self.low_storage: self.create_chkpt(epoch)
        self.tbwriter.close() # close tensorboard writer

    def test_one_batch(self, databatch):
        if isinstance(databatch, list):
            X, y_true = databatch
            X, y_true = X.to(self.device), y_true.to(self.device)
            y_pred = self.net(X)
        else:
            if self.LN not in databatch.keys():
                X = databatch[self.DN].to(self.device)
                y_pred = self.net(X)
                if self.device.type != 'cpu': return y_pred.detach().cpu()
                else: return y_pred
            else:
                X = databatch[self.DN].to(self.device)
                y_true = databatch[self.LN].to(self.device)
                y_pred = self.net(X)
        # return preds and true labels for evaluation
        if self.metrics == None: return None
        else: return {'y_pred':y_pred, 'y_true': y_true}

    def test(self, ret_preds=False):
        self.net.eval()
        test_metrics, y_preds = [], []
        if not isinstance(self.test_dl, list): self.test_dl = [self.test_dl]
        for dl in self.test_dl:
            has_y_true = True
            _test_metrics, _y_preds = {}, []
            for b, databatch in enumerate(progress_bar(dl, total=len(dl))):
                b_ys = self.test_one_batch(databatch)
                if isinstance(b_ys, dict):
                    b_metrics = self.compute_metrics(b_ys)
                    _test_metrics = update_dict(_test_metrics, b_metrics)
                    if ret_preds: _y_preds.append(b_ys)
                else:
                    has_y_true = False
                    if ret_preds: _y_preds.append(b_ys)
            if has_y_true: test_metrics.append({k:np.mean(_test_metrics[k]) for k in _test_metrics})
            if ret_preds: y_preds.append(_y_preds)
        if ret_preds:
            if len(self.test_dl) == 1: return y_preds[0], test_metrics[0]
            else: return y_preds, test_metrics
        else:
            if len(self.test_dl) == 1: return test_metrics[0]
            else: return test_metrics

    def create_chkpt(self, epoch:int, filepath=None):
        savethese = {'model_state': copy.deepcopy(self.net.state_dict()), 'optimizer_state': self.optimizer.state_dict(), 'epoch': epoch}
        if hasattr(self, 'batch_lr_scheduler'): savethese['scheduler_state'] = self.batch_lr_scheduler.state_dict()
        if filepath: _path = filepath
        else: _path = self.chkptdir/self.chkpt_filename
        torch.save(savethese, _path)

    def load_chkpt(self, filepath=None, strict=True):
        if filepath: _path = filepath
        else: _path = self.chkptdir/self.chkpt_filename
        loaded = torch.load(_path)
        self.net.load_state_dict(loaded['model_state'], strict=strict)
        self.optimizer.load_state_dict(loaded['optimizer_state'])
        if hasattr(self, 'batch_lr_scheduler'): self.batch_lr_scheduler.load_state_dict(loaded['scheduler_state'])
        self.init_epoch = loaded['epoch']

    def save_weights(self, filepath=None):
        if filepath: _path = filepath
        else: _path = self.chkptdir/self.wts_filename
        torch.save(copy.deepcopy(self.net.state_dict()), _path)

    def load_weights(self, filepath=None, strict=True):
        if filepath: _path = filepath
        else: _path = self.chkptdir/self.wts_filename
        self.net.load_state_dict(torch.load(_path), strict=strict)

    def compute_metrics(self, ys, threshold=0.5):
        """metrics to evaluate network training"""
        if ys == None: return None
        y_pred, y_true  = ys["y_pred"], ys["y_true"]

        if y_true.shape == y_pred.shape:
            y_true = y_true.detach().cpu().numpy().flatten()
            y_pred = y_pred.detach().cpu().numpy().flatten()
            if "auc" in self.metrics.metrics:
                y_probs = ys["y_probs"].detach().cpu().numpy().flatten()
                return self.metrics(y_true, y_pred, y_probs)
            else: return self.metrics(y_true, y_pred)
        elif y_pred.shape[-1] == self.num_cls:
            y_pred = y_pred.argmax(dim=1)
            y_pred = y_pred.detach().cpu().numpy().flatten()
            y_true = y_true.detach().cpu().numpy().flatten()
            return self.metrics(y_true, y_pred)
        else: raise ValueError(f"Problem in {self.__name__}.compute_metrics method. Check y_pred and y_true.")
