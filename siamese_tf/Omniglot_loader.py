import os

import numpy as np
from matplotlib.pyplot import imread


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

    # we load every alphabet separately so we can isolate them later
    for alphabet in os.listdir(path):
        print("loading alphabet: {}".format(alphabet))
        lang_dict[alphabet] = [curr_y, None]
        alphabet_path = os.path.join(path, alphabet)

        # every letter/category has it's own column in the array, so load separately
        for letter in os.listdir(alphabet_path):
            cat_dict[curr_y] = (alphabet, letter)
            category_images = []
            letter_path = os.path.join(alphabet_path, letter)

            # read all the images in the current category
            for filename in os.listdir(letter_path):
                image_path = os.path.join(letter_path, filename)
                image = imread(image_path)
                category_images.append(image)
                y.append(curr_y)

            try:
                X.append(np.stack(category_images))

            # edge case - last one
            except ValueError as e:
                print("error - category_images: {}".format(category_images))

            lang_dict[alphabet][1] = curr_y
            curr_y += 1

    y = np.vstack(y)
    X = np.stack(X)
    return X, y, lang_dict


def main():
    root_path = os.path.dirname(os.path.dirname(__file__))
    siamese_dir = os.path.join(root_path, "siamese_tf")
    omniglot_dir = os.path.join(siamese_dir, "omniglot")

    train_folder = os.path.join(omniglot_dir, "images_background")
    print("process train data: {}".format(train_folder))
    X, y, c = omniglot_loader(train_folder)

    test_folder = os.path.join(omniglot_dir, "images_evaluation")
    print("process test data: {}".format(test_folder))
    omniglot_loader(test_folder)


if __name__ == '__main__':
    main()
