from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import os
from urllib.request import urlretrieve
import sys
import zipfile
import tarfile
import pickle
import cv2 as cv
import matplotlib.pyplot as plt

def print_download_progress(count, block_size, total_size):
    pct_complete = float(count * block_size)/total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()



def download_cifa10(output_dir=None, url=None):
    if url == None:
        url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    if output_dir == None:
        output_dir = "./cifa10"

    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)

    file_path = url.split("/")[-1]
    file_path = os.path.join(output_dir, file_path)

    if os.path.exists(file_path) is False:
        file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=print_download_progress)


    print("")
    print("Download finished, Extracting file")

    if file_path.endswith(".zip"):
        zipfile.ZipFile(file=file_path, mode="r").extractall(output_dir)

    elif file_path.endswith((".tar.gz", ".tgz")):
        tarfile.open(name=file_path, mode="r:gz").extractall(output_dir)

def make_data(dir=None):
    if dir == None:
        return None
    else:
        f = os.listdir(dir)

        for i in range(5):
            file_name = os.path.join(dir, "data_batch_" + str(i+1))
            f = open(file_name, 'rb')
            data_dict = pickle.load(f, encoding='latin1')
            f.close()

            _X = data_dict["data"]

            _Y = data_dict["labels"]

            _X = _X.reshape([-1, 3, 32, 32])

            plt.imshow(_X[0,1,:,:])
            plt.show()
            print("dkjflkdj")


if __name__ == '__main__':
    download_cifa10()
    make_data("./cifa10/cifar-10-batches-py")