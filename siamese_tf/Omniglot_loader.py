import numpy as np
import tensorflow as tf

def omniglot_loader(path, n=0):
    """

    :param path:
    :param n:
    :return:
    """
    X = []
    y = []
    cat_dict = {}
    lang_dict = {}
    curr_y = n

    # we load every alphabet seperately so we can isolate them later
