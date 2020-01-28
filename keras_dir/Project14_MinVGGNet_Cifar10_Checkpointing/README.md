### Project 14- Checkpointing Models

<p align="center">
    <img width="800" height="400"
     src="https://images.unsplash.com/photo-1474546652694-a33dd8161d66?ixlib=rb-0.3.5&q=80&fm=jpg&crop=entropy&cs=tinysrgb&w=1080&fit=max&ixid=eyJhcHBfaWQiOjExNzczfQ&s=62a31a4a0ec39ecda178464988df6e22">
</p>

Description:

This repository contains Python source code that implements an image
 classification software by training the MinVGGNet CNN architecture on the
  Cifar10 dataset. However, the primary aim of this project is the
   demonstration of model checkpointing that allows to save a model and
    its weights to the disk during
    the training process as hdf5 files. In its current implementation, a
     checkpoint file is created and saved on those epochs of the training
      process when the validation loss decreases.
       

Technology Used:

* Python 3

Libraries Used:

* numpy
* seaborn
* sklearn
* Keras
