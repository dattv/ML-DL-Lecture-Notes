import numpy as np
import cv2 as cv
import os

random_byte_array = bytearray(os.urandom(120000))
flat_numpy_array = np.asarray(random_byte_array)

# Convert the array to 400x300
gray_image = flat_numpy_array.reshape(300, 400)
cv.imwrite("gray_400x300.jpg", gray_image)

# Convert the array to 400x100x3
color_image = flat_numpy_array.reshape(100, 400, 3)
cv.imwrite("color_400x100.jpg", color_image)
