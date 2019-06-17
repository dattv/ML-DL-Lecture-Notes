# Siamese Neural Network
This tutorial to learn how to develop Siamese Neural network with <b>Tensorflow framework</b>.
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

1. In a short while we will see that to train this network, we do not need too many instances of a class ans only few are enought to build a good model.

2. <b> Biggest advantages</b> is that let's say in case of face recognition, we have a new employee who has joined the organization. Now inorder for the network to detect his face, we only require a single of afew images of his face which will be stored in the database. Using this as the reference image, the network will calculate the similarity for any new instance presented to it. Thus we say that network predicts the score in one short.

### Application

1. Face recognition [2]
   
2. Drug discovery [3]

3. Offline signature verification system [4] 

### Dataset

Omniglot dataset:
1623 hand drawn characters from 50 different alphabets. Every character there are just 20 examples. each image have size of 105x105.
All of the data file is stored in the [link](https://github.com/brendenlake/omniglot/tree/master/python) 
. We download two file [image_background.zip](https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip) and [images_eveluation.zip](https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip).
 
[code_example_download_omniglot_dataset](https://github.com/dattv/ML-DL-Lecture-Notes/blob/master/siamese_tf/Omniglot_dataset.py)

[code_example_load_data_to_tensor](https://github.com/dattv/ML-DL-Lecture-Notes/blob/master/siamese_tf/Omniglot_loader.py)

output of load_data_to_tensor function are X, y, c
1. X are images
2. y index of character
3. c are label

### Mapping the problem to binary classification task
 
We map this problem into classification problem, (we need {X, Y} := {input, target} data type)

input := Pair of images

target := {1 if both contain the same character, 0 if both images contain different class}
 
![image](https://user-images.githubusercontent.com/29138292/59565345-8d270580-907c-11e9-98ae-d09d76218f1c.png)

Thus we need to create pairs of images along with the target variable, as shown above, to be fed as input to the Siamese Network. Note that even though characters from Sanskrit alphabet are shown above, but in practice we will generate pairs randomly from all the alphabets in the training data.

### Siamese model
![image](https://user-images.githubusercontent.com/29138292/59574945-79fb5080-90e3-11e9-977f-73ff06481531.png)

![image](https://user-images.githubusercontent.com/29138292/59575004-c21a7300-90e3-11e9-812e-1f02f1c60fc1.png)

[code_example_siamese_model](https://github.com/dattv/ML-DL-Lecture-Notes/blob/master/siamese_tf/siamese_model.py)


![image](https://user-images.githubusercontent.com/29138292/59576476-29d3bc80-90ea-11e9-9bac-04530f94e404.png)

![image](https://user-images.githubusercontent.com/29138292/59588417-bba2f000-9111-11e9-9a2d-b3bbdf9d1b50.png)

# Reference

[1] [One Shot Learning with Siamese Networks using Keras](https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d)

[2] [https://www.youtube.com/watch?v=wr4rx0Spihs](https://www.youtube.com/watch?v=wr4rx0Spihs)

[3] [One-shot Learning Methods Applied to Drug Discovery with DeepChem](https://www.microway.com/hpc-tech-tips/one-shot-learning-methods-applied-drug-discovery-deepchem/)

[4] [SigNet: Convolutional Siamese Network for Writer Independent Offline Signature Verification](https://arxiv.org/abs/1707.02131)

[5] [One-Shot-Learning-with-Siamese-Networks](https://github.com/hlamba28/One-Shot-Learning-with-Siamese-Networks)