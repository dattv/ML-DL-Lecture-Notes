# Understanding Deep Learning for Computer Vision

Computer vision as a field has a long history. With the emergence of deep learning, computer vision has proven to be useful for various applications. Deep learning is a collection of techniques from <b>artificial neural network (ANN)</b>, which is a branch of machine learning. ANNs are modelled on the human brain; there are nodes linked to each other that pass information to each other.

## Perceptron
An artificla neuron of perceptron takes several inputs and preforms a weighted summation to produce an output. The weights of the perceptron is determined during the training process and is based on the training data. 

## Activation functions
The activation functions make neural nets nonlinear. An activation function decides whether a perceptron should fire or not. During training activation, functions play an important role in adjusting the gradients. An activation function such as sigmoid, This nonlinear behaviour of the activation function gives the deep nets to learn complex functions. Most of the activation functions are continuous and differential functions, except rectified unit at 0. A continuous function has small changes in output for every small change in input. A differential function has a derivative existing at every point in the domain.
In order to train a neural network, the function has to be differentiable. Following are a few activation functions.

* [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) can be considered a smoothened step function and hence differentiable. Sigmoid is useful for converting any value to probabilities and can be used for binary classification. The sigmoid maps input to a value in the range of 0 to 1. After some learning, the change may be small. Another activation function called tanh, explained in next section, is a scaled version of sigmoid and avoids the problem of a vanishing gradient.

* [Tangent]() The hyperbolic tangent function, or tanh, is the scaled version of sigmoid. Like sigmoid, it is smooth and differentiable. The tanh maps input to a value in the range of -1 to 1. The gradients are more stable than sigmoid and hence have fewer vanishing gradient problems. Both sigmoid and tanh fire all the time, making the ANN really heavy. The Rectified Linear Unit (ReLU) activation function, explained in the next section, avoids this pitfall by not firing at times

* [ReLu]() ReLu can let big numbers pass through. This makes a few neurons stale and they don't fire. This increases the sparsity, and hence, it is good. The ReLU maps input x to max (0, x), that is, they map negative inputs to 0, and positive inputs are output without any change. Because ReLU doesn't fire all the time, it can be trained faster. Since the function is simple, it is computationally the least expensive. Choosing the activation function is very dependent on the application. Nevertheless, ReLU works well for a large range of problems.

# Artificial nerual network (ANN)


# Reference 
[1] [Deep Learning for Computer Vision]()