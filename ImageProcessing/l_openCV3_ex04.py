# Reading and Writing Video
import numpy as np
import cv2 as cv
import os

from numpy.ma.extras import _fromnxfunction_allargs
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
video_url = "http://www.cvc.uab.es/~bagdanov/master/videos/car-overhead-1.avi"
PATH = os.getcwd()
video_folder = os.path.join(PATH, "video")
if os.path.isdir(video_folder) == False:
    os.mkdir(video_folder)

    with tqdm(unit="B", unit_scale=True, leave=True, miniters=1, desc=video_folder.split("/")[-1]) as t:
        urllib.request.urlretrieve(video_url, filename=os.path.join(video_folder, "video01.avi"), reporthook=my_hook(t), data=None)

video_capture = cv.VideoCapture(os.path.join(video_folder, "video01.avi"))
fps = video_capture.get(cv.CAP_PROP_FPS)
size = (int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH)),
        int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT)))

video_writer = cv.VideoWriter("myout.avi",
                              cv.VideoWriter_fourcc("I", "4", "2", "0"), fps, size)

success, frame = video_capture.read()
while success:
    video_writer.write(frame)
    success, frame = video_capture.read()

video_capture.release()


clicked = False
def onMouse(event, x, y, flags, param):
    global  clicked
    if event == cv.EVENT_LBUTTONUP:
        clicked = True

video_capture = cv.VideoCapture(os.path.join(video_folder, "video01.avi"))
cv.namedWindow("my_window")
cv.setMouseCallback("my_window", onMouse)
success, frame = video_capture.read()

while success and cv.waitKey(1) == -1 and not clicked:
    cv.imshow("my_window", frame)
    success, frame = video_capture.read()

cv.destroyAllWindows("my_window")
video_capture.release()



