### Project 12- MinVGGNet Performance on Cifar-10 dataset

<p align="center">
    <img width="800" height="400"
     src="https://samyzaf.com/ML/cifar10/cifar1.jpg">
</p>

Description:

This repository contains source code, implementing a Python coded
CNN based image classification software, that classifies 10 classes of target
images using the Keras library. The primary goal of this project is to
 implement a very simple, `MinVGGNet` architecture 
    trained on the `Cifar-10` dataset to
     differentiate between 10 classes of targets. 
      The associated Jupyter file `minvggnet_cifar10`shows a cursory
       implementation of the
       model training, testing and prediction.

A normalisation regime consisting of periodic Batch Normalization and
 implementation of Dropout normalisation with 0.5 keep probability allowed
  the attainment of around 80% validation accuracy on the dataset with the
   `MinVGGNet` architecture.
   
A second Jupyter file `minvggnet_cifar10_ler_decay` is also provided that
 showcases the implementation of custom learning rate schedulers within the
  Keras library and the implementation of custom step wise learning rate
   schedules with 2 decay rate factor values on the aformentioned dataset. I
   It was observed tht, aggressive learning rate decays led to decreased
    overall testing accuracy and early model training saturation.   
       

Technology Used:

* Python 3

Libraries Used:

* numpy
* seaborn
* sklearn
* Keras
