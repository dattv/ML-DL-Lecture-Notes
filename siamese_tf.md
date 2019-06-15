# Siamese Neural Network
This tutorial to learn how to develop Siamese Neural network with Tensorflow framework.
This is converted from [1]

## Introduction
Assume that we want to build face recognition system for a small organization with only 10 employees.
if we use traditional classification apporach, we might come up with a system 

![image](https://user-images.githubusercontent.com/29138292/59548024-06d2cc80-8f73-11e9-90a8-bc79a737bde2.png)

### Problems
1. Tro train such a system, we need alot of different images of each of the 10 person in the organization.

1. What if a new person joins or leaves the organization? you need to take the pain of collecting data again an dre-train the entire model again. This practically not possible specially for large organizations where recruitment and attrition in happening almost every week.

### Advantages of One-short learning
:+1: Require only one or afew training example for each class.

### Solution
one short classification which helps to solve both problems above.


![image](https://user-images.githubusercontent.com/29138292/59548083-aa23e180-8f73-11e9-8a31-2d74ce198354.png)



# Reference

[1] [One Shot Learning with Siamese Networks using Keras](https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d)