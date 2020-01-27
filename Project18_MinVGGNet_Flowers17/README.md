### Project 18- Effect of Data Augmentation on CNN 

<p align="center">
    <img width="800" height="400"
     src="https://dpzbhybb2pdcj.cloudfront.net/elgendy/v-7/Figures/Img_01-04A_227.jpg">
</p>

Description:

This repository contains 2 Jupyter files that serve as source codes
 demonstrating the effect of Data Augmentation on CNN training accuracy. A
  MinVGGNet CNN architecture was trained on a dataset containing 17
   categories of flower species:
   
   <p align="center">
    <img width="500" height="400"
     src="./flowers17.JPG">
</p>

Owing to the lower number of training samples per class of flower species, a
 low accuracy of around 60% was obtained on the un-altered dataset with a
  large degree of train data overfitting shown by the 100% training accuracy
   below:



<p align="center">
    <img width="600" height="300"
     src="./simple.JPG">
</p>

Subsequently, data augmentation routine was employed on the dataset to
 impose random rotation, width and height shifts, shear distortion and zoom
  to allow the model to generalise better on test data. Consequently, an
   improved test accuracy of around 75% was obtained as can be seen through
    the training curve on the augmented dataset below:
    
 <p align="center">
    <img width="600" height="300"
     src="./augmented.JPG">
</p>

Technology Used:

* Python 3

Libraries Used:

* numpy
* seaborn
* Keras
* OpenCV
