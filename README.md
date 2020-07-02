# Yet Another Machine Learning Framework (YAMLF)

(pre-release)

YAMLF is a PyTorch based light-weight Supervised and Semi-Supervised Machine Learning model training and evaluation Python 3 module. It provides helper functions and classes for easy management of the training procedure. It provides the fundamental tools that we write again and again for every model training like training loop, evaluation, setting schedulers, recording loss and metrics, etc. This module sits inbetween PyTorch's flexible but long code process and Keras' high-level api. It's as flexible as PyTorch but eliminates the code redundancy. There are 5 steps to train a model - set hyperparameters, make dataset class, create dataloaders, make model, and train. Each step is replaceable with plain PyTorch.

How to install:

`pip install yamlf`


How to use:

Example Notebooks are given in `tutorials` folder.
