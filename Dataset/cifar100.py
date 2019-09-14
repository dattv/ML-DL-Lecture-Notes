import os
import numpy as np
from tqdm import tqdm
import tarfile
from urllib.request import urlretrieve
import pickle


def my_hook(t):
    """
    Wraps tqdm instance
    :param t:
    :return:
    """
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """

        :param b:       int option
        :param bsize:   int
        :param tsize:
        :return:
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


class cifar100:
    def __init__(self, cifar100_dir=None):
        self.name = "Cifar100 dataset"
        self.n_channels = 3
        self.width = 32
        self.heigh = 32
        self.DIR = "./cifar100_dataset"
        if os.path.isdir(self.DIR) == False:
            os.mkdir(self.DIR)

        cifar100_url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

        name = cifar100_url.split("/")
        name = name[len(name) - 1]

        full_file_path = os.path.join(self.DIR, name)

        if os.path.isfile(full_file_path) == False:
            print("downloading from: {}".format(cifar100_url))
            with tqdm(unit="B", unit_scale=True, leave=True, miniters=1, desc=cifar100_url.split("/")[-1]) as t:
                urlretrieve(cifar100_url, filename=full_file_path, reporthook=my_hook(t), data=None)
            print("finish download")

        # extract compressed file
        tar = tarfile.open(full_file_path)
        sub_folders = tar.getnames()
        train_subfolder = [f for f in sub_folders if os.path.split(f)[1] == "train"]
        self._train_subfolder = train_subfolder[0]
        test_subfolder = [f for f in sub_folders if os.path.split(f)[1] == "test"]
        self._test_subfolder = test_subfolder[0]
        tar.extractall()
        tar.close()

    def get_data(self):

        with open(self._train_subfolder, 'rb') as fo:
            try:
                self.samples_train = pickle.load(fo)
            except UnicodeDecodeError:  # python 3.x
                fo.seek(0)
                self.samples_train = pickle.load(fo, encoding='latin1')

        with open(self._test_subfolder, 'rb') as fo:
            try:
                self.samples_test = pickle.load(fo)
            except UnicodeDecodeError:
                fo.seek(0)
                self.samples_test = pickle.load(fo, encoding='latin1')

        train_data_img = self.samples_train["data"]
        train_data_labels = self.samples_train["fine_labels"]
        test_data_img = self.samples_test["data"]
        test_data_lables = self.samples_test["fine_labels"]
        self.num_class = len(train_data_labels[0])

        return (train_data_img, train_data_labels), (test_data_img, test_data_lables)

dataset = cifar100()
(train_data_img, train_data_labels), (test_data_img, test_data_labels) = dataset.get_data()
train_img_number = len(train_data_img)
test_img_number = len(test_data_img)
print(dataset.name)
print("Number of Images in train set: {}".format(train_img_number))
print("Number of Images in test set: {}".format(test_img_number))
print("{}".format(train_data_img.shape))
print("Images size: {}x{}x{}".format(dataset.heigh, dataset.width, dataset.n_channels))

train_data_labels_one_hot = np.zeros((train_img_number, dataset.num_class))
train_data_labels_one_hot[np.arange(train_img_number), train_data_labels] = 1

test_data_labels_one_hot = np.zeros((test_img_number, dataset.num_class))
test_data_labels_one_hot[np.arange(test_img_number), test_data_labels] = 1
