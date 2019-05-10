# CIFAR10

import tensorflow as tf
import os

from tqdm import tqdm
import urllib

CIFAR10_DIR = "./CIFA"
if os.path.isdir(CIFAR10_DIR) == False:
    os.mkdir(CIFAR10_DIR)

response = urllib.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
