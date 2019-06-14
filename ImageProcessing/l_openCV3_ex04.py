# Reading and Writing Video
import numpy as np
import cv2 as cv
import os
from tqdm import tqdm
import urllib.request


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


# download video
video_url = "http://www.ee.cuhk.edu.hk/~xgwang/filtTrk_parkinglot.mat"
PATH = os.getcwd()
video_folder = os.path.join(PATH, "video")
if os.path.isdir(video_folder) == False:
    os.mkdir(video_folder)

    with tqdm(unit="B", unit_scale=True, leave=True, miniters=1, desc=video_folder.split("/")[-1]) as t:
        urllib.request.urlretrieve(video_url, filename=os.path.join(video_folder, "video01"), reporthook=my_hook(t), data=None)
