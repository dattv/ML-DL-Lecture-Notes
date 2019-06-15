from builtins import property

import cv2 as cv
import numpy as np
import time
import os

class capture_manager(object):
    def __init__(self, capture, previewWindowManager=None, shouldMirrorPreview=False):
        self.previewWindowManager = previewWindowManager
        self.shouldMirrorPreview = shouldMirrorPreview

        self._capture = capture
        self._channel = 0

        self._enteredFrame = False
        self._frame = None
        self._imageFileNmae = None
        self._videoFileName = None
        self._videoEncoding = None
        self._videoWrite = None

        self._startTime = None
        self._framesElapsed = 0
        self._fpsEstimate = None
        self._fpsEstimate = None

    @property
    def chanel(self):
        return self._channel

    @property
    def chanel(self):
        return self._channel

    @chanel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retieve()
            return self._frame

    @property
    def isWritingImage(self):
        return self._videoFileName is not None

    def enterFrame(self):
        """
        Capture the next frame, if any
        :return:
        """

        assert not self._enteredFrame, "previous enterFrame() has no matching exitFrame()"

        if self._capture is not None:
            self._enteredFrame = self._capture.grab()

    def exitFrame(self):
        """
        Draw to the window, write to files, release the frame
        :return:
        """
