"""
Image related functions and classes
Author: Jitender Singh Virk [virksaab.github.io]
Last updated: 29 Jun 2020
"""

import torch, PIL
import torch.nn as nn
import torchvision as tv
import albumentations as alb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from albumentations.pytorch.transforms import ToTensorV2 as ToTensor
from typing import Tuple, List
from easydict import EasyDict

# This monkey-patch is there to be able to plot tensors (from fastai)
torch.Tensor.ndim = property(lambda x: len(x.shape))
plt.style.use('ggplot')

# For data Normalization
cifar_stats = ([0.491, 0.482, 0.447], [0.247, 0.243, 0.261])
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

############################### DATASET AND DATALOADER ####################
class InfDL(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an infinite iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


class LoadData:
    """
    Create dataloaders for datasets.
    Args:
        trainset(Dataset or list of Dataset class): train dataset class
        valset(Dataset or list of Dataset class): val dataset class
        testset(Dataset or list of Dataset class): test dataset class
        collate_fn (Callable): batch making function
        defs(EasyDict or dict): default settings dict
    """
    def __init__(self,
        trainset:Tuple[torch.utils.data.Dataset, List[torch.utils.data.Dataset]],
        valset:Tuple[torch.utils.data.Dataset, List[torch.utils.data.Dataset]]=None,
        testset:Tuple[torch.utils.data.Dataset, List[torch.utils.data.Dataset]]=None,
        shuffle:bool=True,
        collate_fn=None,
        defs:Tuple[EasyDict, dict]=None
    ):

        if defs == None: raise AttributeError('Please pass default settings dict as defs=<DefaultSettings>')
        self.__dict__.update(defs)
        # dataloaders
        # trainset
        if isinstance(trainset, list): self.train_dl = [InfDL(dataset=ds, batch_size=self.batchsize, shuffle=shuffle, num_workers=self.num_workers, collate_fn=collate_fn) for ds in trainset]
        else: self.train_dl = InfDL(dataset=trainset, batch_size=self.batchsize, shuffle=shuffle, num_workers=self.num_workers, collate_fn=collate_fn)
        # valset
        if valset != None:
            if isinstance(valset, list): self.val_dl = [torch.utils.data.DataLoader(ds, self.batchsize, num_workers=self.num_workers, collate_fn=collate_fn) for ds in valset]
            else: self.val_dl = torch.utils.data.DataLoader(valset, self.batchsize, num_workers=self.num_workers, collate_fn=collate_fn)
        # testset
        if testset != None:
            if isinstance(testset, list): self.test_dl = [torch.utils.data.DataLoader(ds, self.batchsize, num_workers=self.num_workers, collate_fn=collate_fn) for ds in testset]
            else: self.test_dl = torch.utils.data.DataLoader(testset, self.batchsize, num_workers=self.num_workers, collate_fn=collate_fn)

    def show_sample(self, sample_from='train', figsize=None):
        _idx = random.randint(0, len(eval(f"self.{sample_from}set")))
        _sample = eval(f"self.{sample_from}set")[_idx]
        X = _sample[self.DN]
        if self.LN in _sample.keys(): y_true = _sample[self.LN]
        else: y_true = None
        if figsize: plt.figure(figsize=figsize)
        plt.imshow(X.permute(1,2,0))
        if y_true:
            if hasattr(self, 'ltoc'): plt.title(str(self.ltoc[y_true]))
            else: plt.title(str(y_true))
        plt.axis('off')
        plt.show()

    def show_batch(self, cols:int=None, batch_from='train'):
        _samples = next(eval(f"self.{batch_from}_dl"))
        inputs = _samples[self.DN]
        if self.LN in _samples.keys():
            labels = _samples[self.LN]
        print('Single Sample shape:', inputs[0].shape)
        _bs = inputs.size(0)
        if cols == None:
            if _bs <= 8:
                cols, rows = 8, 1
            else:
                cols = 16
                rows = _bs//cols
        # print(cols, rows)
        n_flag = True if inputs.max().item() != 1. and inputs.min().item() != 0. else False
        if n_flag: print('scaling inputs to range [0,1].')
        if isinstance(labels, torch.Tensor): labels = [l.item() for l in labels]
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2,(rows*2)+1))
        for i, ax in enumerate(axes.flat):
            sample = inputs[i].cpu()
            if n_flag: sample = minmaxscaler(sample)
            ax.imshow(sample.permute(1,2,0))
            if labels:
                if hasattr(self, 'ltoc'):
                    ax.set_title(self.ltoc[labels[i]])
                else:
                    ax.set_title(str(labels[i]))
            ax.axis('off')
        plt.show()

    def __repr__(self):
        out = "yamlf.vision.LoadData class"
        out += f"\ntrainset -> num samples: {len(self.train_dl.dataset)}, num batches: {len(self.train_dl)}"
        if hasattr(self, 'val_dl'):
            out += f"\nvalset -> num samples: {len(self.val_dl.dataset)}, num batches: {len(self.val_dl)}"
        if hasattr(self, 'test_dl'):
            out += f"\ntestset -> num samples: {len(self.test_dl.dataset)}, num batches: {len(self.test_dl)}"
        return out