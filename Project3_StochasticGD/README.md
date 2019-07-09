### Project 3: Implementation of the Stochastic Gradient Descent Optimizer Algorithm
<p align="center">
    <img width="600" height="300"
     src="https://www.bogotobogo.com/python/scikit-learn/images/Batch-vs-Stochastic-Gradient-Descent/stochastic-vs-batch-gradient-descent.png">
</p>

Description:

This repository contains source code, implementing a Stochastic 
Gradient Descent Algorithm to minimise the cost function of least-squared-loss 
of binary classification of 2 dimensional data being classified via 
the Logistic Regression Algorithm.

The primary goal of the project is to compare the model training 
performances of the 2 optimizers: Stochastic Gradient Descent (SGD)
 and vanilla Gradient Descent previously implemented in [Project2](https://github.com/nombreinvicto/DeepLearningCV/tree/master/Project2_SimpleGradientDescent).

It is to be noted that the current implementation of the 
Stochastic Gradient 
Descent Algorithm minimises the least-square-loss function which is 
not convex. While the cross entropy log loss function is almost 
unanimously used in Logistic Regression tasks due it its convexity, 
the implementation of a non convex loss function in this project has 
been considered for simplicity and to rather cast the focus on how the 
Stochastic Gradient Descent Algorithm really works.

Technology Used:

* Python 3

Libraries Used:

* numpy
* matplotlib
* seaborn