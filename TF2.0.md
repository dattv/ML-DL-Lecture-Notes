# Tensorflow 2.0
This artist for noting some useful thing in using tensorflow 2.0 and python3.6

I collected this from [Tensorflow 2.0 quick start guide]() and [Tensorflow]()

## Tensorflow Ecosystem
1. [eager execution]()

2. [tf.data]() is an API that allows you to build complicated data input pipelines from simpler, reusable parts

3. [tensorflow.js]() is a collection of APIs that allow you to build and train models using either the low-level JavaScript linear algebra library or the high-level layers API. Hence, models can be trained and run in a browser.

4. [tensorflow lite]() is a lightweight version of TensorFlow for mobile and embedded device

5. [tensorflow hub]() s a library designed to foster the publication, discovery, and use of reusable modules of machine learning models. In this context, a module is a self-contained piece of a TensorFlow graph together with its weights and other assets. The module can be reused in different tasks in a method known as transfer learning

6. [tensor board]() is a suite of visualization tools supporting the understanding, debugging, and optimizing of TensorFlow programs

## How to install 
please follow [Install TensorFlow-gpu 2.0 on Anaconda for Windows 10/Ubuntu](https://medium.com/@shaolinkhoa/install-tensorflow-gpu-2-0-alpha-on-anaconda-for-windows-10-ubuntu-ced099010b21).

Basically, It's required conda => python 3.6, cudnn, cupti, cudatoolkit10.0 => tensorflow-gpu

## Data to ANN
Three important ways of constructing a data pipeline:
1. from in-memory NumPy

2. from Comma-Seperated-Value (CSV)

3. TFRecord

## Batch normalization layer and dropout layer
Batch normalization is a layer that takes its inputs and outputs the same number of outputs with activations that have zero mean and unit variance, as this has been found to be beneficial to learning. Batch normalization regulates the activations so that they neither become vanishingly small nor explosively big, both of which situations prevent the network from learning.

Dropout layer is a layer where a certain percentage of the neurons are randomly turned off during training (not during inference). This forces the network to become better at generalizing since individual neurons are discouraged from becoming specialized with respect to their inputs.




