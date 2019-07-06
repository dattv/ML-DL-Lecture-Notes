import tensorflow as tf
from urllib.request import urlretrieve

from tqdm import tqdm

import os

inception_net_url = "https://github.com/google/inception/blob/master/inception_tensors.pb"

def my_hook(t):
    last_b = [0]
    def inner(b=1, bsize=1, tsize=None):
