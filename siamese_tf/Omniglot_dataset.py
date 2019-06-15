import urllib.request
import numpy as np
import os
from tqdm import tqdm
import zipfile

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

root_path = os.path.dirname(os.path.dirname(__file__))
siamese_folder = os.path.join(root_path, "siamese_tf")

image_background_url = "https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip"
images_evaluation_url = "https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip"

omniglot_folder = os.path.join(siamese_folder, "omniglot")
if os.path.isdir(omniglot_folder) == False:
    os.mkdir(omniglot_folder)

# Download dataset
image_back_ground_file_name = os.path.split(image_background_url)[1]
image_background_full_file_path = os.path.join(omniglot_folder, image_back_ground_file_name)
with tqdm(unit="B", unit_scale=True, leave=True, miniters=1, desc=image_background_url.split("/")[-1]) as t:
    urllib.request.urlretrieve(image_background_url, filename=image_background_full_file_path, reporthook=my_hook(t), data=None)

zip_ref = zipfile.ZipFile(image_background_full_file_path, "r")
zip_ref.extractall(omniglot_folder)
zip_ref.close()

images_evaluation_file_name = os.path.split(images_evaluation_url)[1]
images_evalutaion_full_file_path = os.path.join(omniglot_folder, images_evaluation_file_name)
with tqdm(unit="B", unit_scale=True, leave=True, miniters=1, desc=images_evaluation_url.split("/")[-1]) as t:
    urllib.request.urlretrieve(images_evaluation_url, filename=images_evalutaion_full_file_path, reporthook=my_hook(t), data=None)

zip_ref = zipfile.ZipFile(images_evalutaion_full_file_path, "r")
zip_ref.extractall(omniglot_folder)
zip_ref.close()

