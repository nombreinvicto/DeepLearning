### Project 8- Create Your Own Image Classifier

<p align="center">
    <img width="600" height="300"
     src="https://www.researchgate.net/profile/Vinay_Kumar_N2/publication/274096400/figure/fig2/AS:294855780651019@1447310504719/Average-classification-accuracies-obtained-for-5-trials-a-40-Training-b-60.png">
</p>

Description:

This repository contains source code, implementing a Python coded
 image classification software, that classifies flowers from among
  102 classes using Pytorch. The primary goal of this project is to
   implement transfer learning
 and take advantage of previously trained model architectures like
  `Vgg16`, `Densenet121`, `Alexnet` etc trained on the `Imagenet
  ` dataset to classify other things, which in this case is a
   curated dataset of 102 flower species.
   
The associated Jupyter file shows a cursory implementation of the
 model training, transfer learning, testing and prediction. The
  associated `train.py` file is meant to act as a command line app
   that can repeatedly train on different architectures based on
    hyperparameter inputs from the user via the command line. The
     associated `predict.py` file acts as a predictor app that
      takes in a trained checkpoint model and image path via
       command line and subsequently uses the checkpoint model to
        make predictions on the supplied image.
        
The `train.py` file accepts command line arguments which have
 been slightly modified from the exact requirements laid out in the
  rubric. The changes are as follows:
  
  - The training dataset has to be now supplied as a command line
   argument via the `--data_dir` tag.
   
  - The `hidden_units` tag has been changed to be a list of nodes
   in the hidden layers instead of a simple number. So now, it has to
    be supplied as a string encoded list to the command line as
     follows: 
     
     `--hidden_units "[512, 256]"` 
  - The choice of which device (CUDA or CPU) to run the training on
   has been changed from the `--gpu` tag for the command line to
    the `--device` tag which requires an argument. Therefore, if
     you want to run the training on the GPU, this argument has to
      be supplied as `--device cuda`
      
  - Currently only 3 architectures are implemented, namely `alexnet
  `, `densenet121` and `vgg16`.  
  
  Example input:
  
  `python3 train.py --device cuda --arch vgg16 --learning_rate 0.001 --hidden_units "[512, 256]" --epochs 3`  

In a similar fashion the `predict.py` file does not strictly
 conform to guidelines laid out in the project instructions.
 
   - The choice of which device (CUDA or CPU) to run the prediction on
   has been changed from the `--gpu` tag for the command line to
    the `--device` tag which requires an argument. Therefore, if
     you want to run the training on the GPU, this argument has to
      be supplied as `--device cuda`
      
   - The `--image` and `--checkpoint` command line arguments have
    been converted to required arguments and these represent full
     paths to the image and the checkpoint respectively.
     
   - The prediction result is returned as a dictionary of the top n
    classes of prediction as chosen by the user, where the keys are
     the class names and values are dictionaries of respective
      class labels and class probabilities. 
      
  Example input:
  
  `python3 predict.py --image /home/mhasan3/Desktop
  /ImageClassifierProject/flowers/test/17/image_03911.jpg --checkpoint /home/mhasan3/checkpoint.pth --topk 3 --device cuda`        
          

Technology Used:

* Python 3

Libraries Used:

* json
* argparse
* numpy
* matplotlib
* seaborn
* Pytorch
* PIL














   `     
     
     
       
 
 
      
       