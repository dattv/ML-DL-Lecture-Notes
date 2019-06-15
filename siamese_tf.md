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

Instead of directly classifying an input(test) image to one of th e10 people in the organization, this network instead takes an extra reference image of the person as input and will produce a similarity score denoting the chances tha tthe two input images belong to the same person. Typically the similarity score is squished between 0 and 1 using a sigmoid function; Wherein 0 denotes no similarity and 1 denotes full similarity. Any number between 0 and 1 is interpreted accordingly.

<b>Notice: Thjsi network is not learnign to classify and image directly to any of the output classes. Rather, it is learning a similarity functiono, which takes two images as input and expresses how similar they are. </b>

How does this solve the two problems above:
1. 

# Reference

[1] [One Shot Learning with Siamese Networks using Keras](https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d)