from Omniglot_loader import omniglot_loader

import numpy as np
import os


def get_batch(batch_size, s="train"):
    """
    Create batch of n pair, half same class, half different class
    :param batch_size:
    :param s:
    :return:
    """
    if s == "train":
        X = Xtrain
        categories = train_classes
    else:
        X = Xval
        categories = val_classes

    n_classes, n_examples, w, h = X.shape


def main():
    get_batch(100, "train")


if __name__ == '__main__':
    main()
