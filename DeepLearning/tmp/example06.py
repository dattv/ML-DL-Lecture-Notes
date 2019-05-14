import tensorflow as tf
import numpy as np

import os
import urllib.request

CIFAR_DIR = "./CIFA"

if os.path.isdir(CIFAR_DIR):
    os.mkdir(CIFAR_DIR)

cifar_usr = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
name = cifar_usr.split("/")
name = name[len(name)-1]

urllib.request.urlretrieve(cifar_usr, name)
