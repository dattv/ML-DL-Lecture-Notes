import tensorflow as tf
import numpy as np

import os
import urllib.request

import tarfile
import zipfile
import numpy as np

CIFAR_DIR = "./CIFA"


if os.path.isdir(CIFAR_DIR):
    os.mkdir(CIFAR_DIR)

cifar_usr = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
name = cifar_usr.split("/")
name = name[len(name)-1]

full_file_path = "."+CIFAR_DIR+"/"+name
print("downloading from: {}".format(cifar_usr))
urllib.request.urlretrieve(cifar_usr, full_file_path)
print("finish download")

# extract
if full_file_path.endswith(".tar.gz") or full_file_path.endswith(".tgz"):
    opener, mode = tarfile.open, "r:gz"

to_directory = CIFAR_DIR
os.chdir(to_directory)

print("current working directory: {}".format(os.getcwd()))

try:
    print("try to extract: {}".format(full_file_path))
    file = opener(full_file_path, mode)
    try: file.extractall()
    finally: file.close()
    print("finish extracting")
finally:
    os.chdir(to_directory)



