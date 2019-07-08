import tensorflow as tf
from urllib.request import urlretrieve

from tqdm import tqdm
import tarfile

import os

inception_net_url = "http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz"

def my_hook(t):
    last_b = [0]
    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner

root_path = os.path.dirname(os.path.dirname(__file__))

DeepLearning_path = os.path.join(root_path, "DeepLearning")

inception_file_name = inception_net_url.split("/")
inception_file_name = inception_file_name[len(inception_file_name) - 1]
name = inception_file_name.split(".")[0]

inception_folder = os.path.join(DeepLearning_path, name)
if os.path.isdir(inception_folder) == False:
    os.mkdir(inception_folder)

inception_full_file_path = os.path.join(inception_folder, inception_file_name)

if os.path.isfile(inception_full_file_path) == False:
    print("Downloading {}".format(inception_file_name))
    with tqdm(unit="B", unit_scale=True, leave=True, miniters=1, desc=inception_net_url.split("/")[-1]) as t:
        urlretrieve(inception_net_url, filename=inception_full_file_path, reporthook=my_hook(t), data=None)

tar = tarfile.open(inception_full_file_path)
tar.extractall(path=inception_folder)
tar.close()

from tensorflow.python import pywrap_tensorflow

reader = pywrap_tensorflow.NewCheckpointReader(inception_folder + "/inception_v1.ckpt")
var_to_shape_map = reader.get_variable_to_shape_map()

for name in var_to_shape_map.keys():
    print(name, var_to_shape_map[name])

class inception(object):
    def __init__(self):
        super(inception, self).__init__()

        print("inception")

    def


