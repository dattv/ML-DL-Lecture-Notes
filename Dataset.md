# How to use Dataset in TensorFlow
:-1: ``feed-dict`` is the slowest possible way to pass information to TensorFlow

:+1: the correct way to feed data into your models is to use an input pipeline to ensure that the GPU has never to wait for new stuff to come in.

## Generic Overview
1. Importing Data

2. Create an Iterator

3. Consuming Data

### Importing Data
1. numpy
    1. [dataset_example01]()
    
# References
[1] [Dataset in TensorFlow - towardsdatascience.com](https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428)

[2] [code explaint Dataset in Tensorflow](https://github.com/FrancescoSaverioZuppichini/Tensorflow-Dataset-Tutorial/blob/master/dataset_tutorial.ipynb)