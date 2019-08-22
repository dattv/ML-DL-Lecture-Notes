import os
import sys

from Dataset.download.download import download
import tarfile
import pickle as cPickle
import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def one_hot(vec, vals=10):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out


class CifarLoader(object):
    def __init__(self, source_files):
        self._source = source_files
        self._i = 0
        self.images = None
        self.labels = None

    def load_d(self):
        data = [unpickle(f) for f in self._source]
        images = np.vstack([d["data"] for d in data])
        n = len(images)
        self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1).astype(float) / 255
        self.labels = one_hot(np.hstack([d["labels"] for d in data]), 10)
        return self

    def load_b(self):
        data = [unpickle(f) for f in self._source]
        images = np.vstack([b["data"] for d in data])
        n = len(images)
        self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1).astype(float) / 255
        self.labels = one_hot(np.hstack([b["labels"] for d in data]), 10)
        return self

    def next_batch(self, batch_size):
        x, y = self.images[self._i:self._i + batch_size], self.labels[self._i:self._i + batch_size]
        self._i = (self._i + batch_size) % len(self.images)
        return x, y


class cifar_10:
    def __init__(self, url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", save_folder=None):
        self._url = url

        if save_folder == None:
            self._save_folder = os.path.dirname(sys.modules["__main__"].__file__)
        else:
            self._save_folder = save_folder

        full_file_path = os.path.join(self._save_folder, os.path.split(self._url)[1])

        download(self._url, self._save_folder)

        tar = tarfile.open(full_file_path)
        tar.extractall(path=self._save_folder)
        tar.close()

    def loader(self):

        return CifarLoader(self._save_folder)


class cifar_100:
    def __init__(self, url="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz", save_folder=None):
        self._url = url

        if save_folder == None:
            self._save_folder = os.path.dirname(sys.modules["__main__"].__file__)
        else:
            self._save_folder = save_folder

        full_file_path = os.path.join(self._save_folder, os.path.split(self._url)[1])

        download(self._url, self._save_folder)

        tar = tarfile.open(full_file_path)
        tar.extractall(path=self._save_folder)
        names = tar.getnames()

        tar.close()

        self.train = CifarLoader([os.path.join(self._save_folder, names[0]) + "/train"]).load_d()
        self.test = CifarLoader([os.path.join(self._save_folder, names[0]) + "/test"]).load_d()



if __name__ == '__main__':
    CIFAR_10 = cifar_10(save_folder="../cifar10_data")
    data = CIFAR_10.loader()

