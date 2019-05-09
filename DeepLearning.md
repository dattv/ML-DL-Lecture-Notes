# Understanding Deep Learning for Computer Vision

Computer vision as a field has a long history. With the emergence of deep learning, computer vision has proven to be useful for various applications. Deep learning is a collection of techniques from <b>artificial neural network (ANN)</b>, which is a branch of machine learning. ANNs are modelled on the human brain; there are nodes linked to each other that pass information to each other.

## Perceptron
An artificla neuron of perceptron takes several inputs and preforms a weighted summation to produce an output. The weights of the perceptron is determined during the training process and is based on the training data. 

## Activation functions
The activation functions make neural nets nonlinear. An activation function decides whether a perceptron should fire or not. During training activation, functions play an important role in adjusting the gradients. An activation function such as sigmoid, This nonlinear behaviour of the activation function gives the deep nets to learn complex functions. Most of the activation functions are continuous and differential functions, except rectified unit at 0. A continuous function has small changes in output for every small change in input. A differential function has a derivative existing at every point in the domain.
In order to train a neural network, the function has to be differentiable. Following are a few activation functions.

* [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) can be considered a smoothened step function and hence differentiable. Sigmoid is useful for converting any value to probabilities and can be used for binary classification. The sigmoid maps input to a value in the range of 0 to 1

![picture alt](https://en.wikipedia.org/wiki/Sigmoid_function#/media/File:Logistic-curve.svg "sigmoid: wikipedia")




# Reference 
[1] [Deep Learning for Computer Vision]()