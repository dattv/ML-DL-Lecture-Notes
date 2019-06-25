import urllib.request
import os
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm


def my_hook(t):
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


cifar10_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

_, file_name = os.path.split(cifar10_url)
file_name = file_name.split(".")[0]

root_path = os.path.dirname(os.path.dirname(__file__))
data_set_path = os.path.join(root_path, "Dataset")
data_set_path = os.path.join(data_set_path, "data")

if os.path.isdir(data_set_path) == False:
    os.mkdir(data_set_path)

