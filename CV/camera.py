"""

"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import cv2 as cv
from threading import Thread
from datetime import datetime

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object with dedicated thread.
    """
    def __init__(self, source=0):
        """

        :param source:
        """
        self.stream = cv.VideoCapture(source)
        (self.grabbed, self.frame) = self.stream.read()

        self.stream_fps = self.stream.get(5)
        self.stopped = False

    def start(self):
        """

        :return:
        """
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        """

        :return:
        """
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        """

        :return:
        """
        self.stopped = True

class VideoShow:
    """
    Class that continusly shows a frame using a dedicated thread
    """
    def __init__(self, frame=None, wName='window'):
        """

        :param frame:
        """
        self.frame = frame
        self.stopped = False
        self.wName = wName

    def show(self):
        """

        :return:
        """
        while not self.stopped:
            cv.imshow(self.wName, self.frame)
            if cv.waitKey(1) == ord("q"):
                self.stopped = True
    def start(self):
        """

        :return:
        """
        Thread(target=self.show, args=()).start()
        return self

    def stop(self):
        """

        :return:
        """
        self.stopped = True

if __name__ == '__main__':
    videoGetter = VideoGet(0).start()
    videoShower = VideoShow(videoGetter.frame, wName='window').start()
    it = 0
    frequency = 10000000
    timeElapse = 0
    while True:
        if videoGetter.stopped or videoShower.stopped:
            videoGetter.stop()
            videoShower.stop()
            break
        else:
            first = datetime.now()
            frame = videoGetter.frame
            videoShower.frame = frame
            it += 1
            timeElapse += (datetime.now() - first).total_seconds()
            if it % frequency == 0:
                time = timeElapse / frequency
                print("time elapse: {}, FPS: {}".format(timeElapse, 1000/timeElapse))
                it = 0
                timeElapse = 0


