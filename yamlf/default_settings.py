"""
Default Settings file for machine learning training and testing hyperparameters
Author: Jitender Singh Virk [virksaab.github.io]
Last updated: 29 Jun 2020
"""

from typing import Tuple
from torch.utils.tensorboard import SummaryWriter
import torch, datetime, pathlib

# Default settings for Deep Learning
class DefaultSettings:
    """Easy management of parameters and hyperparameters
        Args:
        datadir: training and testing data folder path
        chkptdir: checkpoints folder path to save logs and weights
        low_storage: if True, don't save tensorboard logs and model weights automatically
        kwargs: additional settings
    """
    def __init__(self, datadir:Tuple[str, pathlib.Path]=None,
     chkptdir:Tuple[str, pathlib.Path]=None, low_storage:bool=False, **kwargs):
        self.low_storage = low_storage
        # dataset class settings
        self.DN = 'inputs'  # inputs name for Neural Network databatch and Dataset class
        self.LN = 'targets' # target column name
        self.scpc = None # samples count per class
        self.ltoc = None # Label to Class name list for classification
        self.ctol = None # Class to labels dict for classification
        # data and paths settings
        self.datadir = datadir if datadir else pathlib.Path('data')
        self.datadir = pathlib.Path(self.datadir)
        if not self.low_storage:
            self.chkptdir = chkptdir if chkptdir else pathlib.Path('chkpts')
            self.chkptdir = pathlib.Path(self.chkptdir)
            self.chkptdir.mkdir(parents=True, exist_ok=True)
            self.chkpt_filename = 'chkpt.pt'
            self.wts_filename = 'wts.pt'
            # Tensorboard settings
            logdir = datetime.datetime.now().strftime("%d%b%Y-%H:%M:%S")
            self.tbwriter = SummaryWriter(log_dir=self.chkptdir/f"tblogs/{logdir}")
        # training and evaluation settings
        self.num_folds = 5 # num of cross validation folds
        self.batchsize = 64
        self.num_workers = 8
        self.epochs = 5
        self.init_epoch = 0
        self.lr = 1e-3 # learning rate
        self.moms = (0.95, 0.85) # optimizer's betas
        self.wd = 0. # weight decay
        self.dropout = 0.1
        # Debugging
        self.train_iters = None
        self.val_iters = None
        # set device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            print('[WARNING] no CUDA GPU detected!')
            self.device = torch.device('cpu')
        # set the passed args
        self.__dict__.update(kwargs)

    @classmethod
    def init(cls, datadir:Tuple[str, pathlib.Path]=None,
    chkptdir:Tuple[str, pathlib.Path]=None, low_storage:bool=False, **kwargs) -> dict:
        return cls(datadir=datadir, chkptdir=chkptdir, low_storage=low_storage, **kwargs).__dict__
