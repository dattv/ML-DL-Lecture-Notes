import os
import sys

from Dataset.download.download import download
import tarfile


class cifar_10:
    def __init__(self, url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", save_folder=None):
        self._url = url

        if save_folder == None:
            self._save_folder = os.path.dirname(sys.modules["__main__"].__file__)
        else:
            self._save_folder = save_folder

        full_file_path = os.path.join(save_folder, os.path.split(self._url)[1])
        list_file = os.listdir(self._save_folder)
        if full_file_path not in list_file:
            download(self._url, self._save_folder)

        tar = tarfile.open(full_file_path)
        tar.extractall()
        tar.close()

class cifar_100:
    def __init__(self, url="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz", save_folder=None):
        self._url = url

        if save_folder == None:
            self._save_folder = os.path.dirname(sys.modules["__main__"].__file__)
        else:
            self._save_folder = save_folder

        full_file_path = os.path.join(save_folder, os.path.split(self._url)[1])
        list_file = os.listdir(self._save_folder)
        if full_file_path not in list_file:
            download(self._url, self._save_folder)

        tar = tarfile.open(full_file_path)
        tar.extractall()
        tar.close()

if __name__ == '__main__':
    CIFAR_10 = cifar_10()
