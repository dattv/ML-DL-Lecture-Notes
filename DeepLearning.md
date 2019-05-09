# Understanding Deep Learning for Computer Vision
<b>This file covers some terminology of DL for CV which is declare in [1].</b>


Computer vision as a field has a long history. With the emergence of deep learning, computer vision has proven to be useful for various applications. Deep learning is a collection of techniques from <b>artificial neural network (ANN)</b>, which is a branch of machine learning. ANNs are modelled on the human brain; there are nodes linked to each other that pass information to each other.

## Perceptron
An artificla neuron of perceptron takes several inputs and preforms a weighted summation to produce an output. The weights of the perceptron is determined during the training process and is based on the training data. 

## Activation functions
The activation functions make neural nets nonlinear. An activation function decides whether a perceptron should fire or not. During training activation, functions play an important role in adjusting the gradients. An activation function such as sigmoid, This nonlinear behaviour of the activation function gives the deep nets to learn complex functions. Most of the activation functions are continuous and differential functions, except rectified unit at 0. A continuous function has small changes in output for every small change in input. A differential function has a derivative existing at every point in the domain.
In order to train a neural network, the function has to be differentiable. Following are a few activation functions.

* [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) can be considered a smoothened step function and hence differentiable. Sigmoid is useful for converting any value to probabilities and can be used for binary classification. The sigmoid maps input to a value in the range of 0 to 1. After some learning, the change may be small. Another activation function called tanh, explained in next section, is a scaled version of sigmoid and avoids the problem of a vanishing gradient.

* [Tangent]() The hyperbolic tangent function, or tanh, is the scaled version of sigmoid. Like sigmoid, it is smooth and differentiable. The tanh maps input to a value in the range of -1 to 1. The gradients are more stable than sigmoid and hence have fewer vanishing gradient problems. Both sigmoid and tanh fire all the time, making the ANN really heavy. The Rectified Linear Unit (ReLU) activation function, explained in the next section, avoids this pitfall by not firing at times

* [ReLu]() ReLu can let big numbers pass through. This makes a few neurons stale and they don't fire. This increases the sparsity, and hence, it is good. The ReLU maps input x to max (0, x), that is, they map negative inputs to 0, and positive inputs are output without any change. Because ReLU doesn't fire all the time, it can be trained faster. Since the function is simple, it is computationally the least expensive. Choosing the activation function is very dependent on the application. Nevertheless, ReLU works well for a large range of problems.

## Artificial nerual network (ANN)
ANN is a collection of perceptrons and activation functions. The perceptrons are connected to form hidden layers or units. The hidden units form the nonlinear basis that maps the input layers to output layers in a lower-dimensional space, which is also called artificial neural networks. ANN is a map from input to output. The map is computed by weighted addition of the inputs with biases. The values of weight and bias values along with the architecture are called model.

The training process determines the values of these weights and biases. The model values are initialized with random values during the beginning of the training. The error is computed using a loss function by contrasting it with the ground truth. Based on the loss computed, the weights are tuned at every step. The training is stopped when the error cannot be further reduced. The training process learns the features during the training. The features are a better representation than the raw images. The following is a diagram of an artificial neural network, or multi-layer perceptron:

Several inputs of x are passed through a hidden layer of perceptrons and summed to the output. The universal approximation theorem suggests that such a neural network can approximate any function. The hidden layer can also be called a dense layer. Every layer can have one of the activation functions described in the previous section. The number of hidden layers and perceptrons can be chosen based on the problem. There are a few more things that make this multilayer perceptron work for multi-class classification problems. A multi-class classification problem tries to discriminate more than ten categories.

# Reference 
[1] [Deep Learning for Computer Vision]()