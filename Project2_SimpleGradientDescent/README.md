### Project 2: Implementation of a Simple Gradient Descent Algorithm on Binary Classification of Data

<p align="center">
  <img width="300" height="150" 
  src="https://cdn-images-1.medium.com/max/1600/1*ZU7P8Y-Ix_pZGzO62Vr9Xg.png">
</p>

Description:

This repository contains source code, implementing a simple 
Gradient Descent Algorithm to minimise the cost function 
of `least-squared-loss` of binary classification 
of 2 dimensional data being classified via the Logistic Regression 
Algorithm. 

It is to be noted that the current implementation of the Gradient 
Descent Algorithm minimises the `least-square-loss` function which 
is not convex. While the cross entropy log loss function is almost 
unanimously used in Logistic Regression tasks due it its convexity,
 the implementation of a non convex loss function in this project 
 has been considered for simplicity and to rather cast the focus on
  how the Gradient Descent Algorithm really works.
  

Technology Used:
    
  * Python 3
  
  Libraries Used:
  * numpy
  * matplotlib
  * seaborn
  * sklearn 