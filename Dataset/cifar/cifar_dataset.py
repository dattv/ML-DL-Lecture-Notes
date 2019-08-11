import os
import sys

import tqdm
from urllib.request import urlretrieve


class cifar_10:
    def __init__(self, url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", save_folder=None):
        self._url = url

        if save_folder == None:
            self._save_folder = os.path.dirname(sys.modules["__main__"].__file__)
        else:
            self._save_folder = save_folder

        


if __name__ == '__main__':
    print("main")
