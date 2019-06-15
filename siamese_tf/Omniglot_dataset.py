import urllib.request
import numpy as np
import cv2 as cv
import os
from tqdm import tqdm

root_path = os.path.dirname(os.path.dirname(__file__))
siamese_folder = os.path.join(root_path, "siamese_tf")

image_background_url = "https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip"
images_eveluation_url = "https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip"

omniglot_folder = os.path.join(siamese_folder, "omniglot")
if os.path.isdir(omniglot_folder) == None:
    os.mkdir(omniglot_folder)

