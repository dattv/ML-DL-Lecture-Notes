import os

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
    for alphabet in os.listdir(path):
        print("loading alphabet: {}".format(alphabet))
        lang_dict[alphabet] = [curr_y, None]
        alphabet_path = os.path.join(path, alphabet)

        # every letter/category has it's own column in the array, so load seperately



def main():
    root_path = os.path.dirname(os.path.dirname(__file__))
    siamese_dir = os.path.join(root_path, "siamese_tf")
    omniglot_dir = os.path.join(siamese_dir, "omniglot")

    train_folder = os.path.join(omniglot_dir, "images_background")

    omniglot_loader(train_folder, 10)


if __name__ == '__main__':
    main()


