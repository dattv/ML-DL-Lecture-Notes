import tensorflow as tf
import os

# Load mnist dataset
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets("MNIST_data", one_hot=True)

train_mnist_data = mnist_data.train
test_mnist_data = mnist_data.test
valid_mnist_data = mnist_data.validation

LOG_DIR = "./tmp"
if os.path.isdir(LOG_DIR) == False:
    os.mkdir(LOG_DIR)

