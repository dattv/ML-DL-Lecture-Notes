import os

import cv2 as cv
import numpy as np

# Reading and writing image file
img = np.zeros((3, 3), dtype=np.uint8)
print(img)
print("img shape: {}".format(img.shape))

img_color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
print(img_color)
print("img_color shape: {}".format(img_color.shape))

root = os.path.dirname(os.path.dirname(__file__))
sample_img_file_path = os.path.join(root, "DeepLearning")
sample_img_file_path = os.path.join(sample_img_file_path, "kagglecatsanddogs_3367a")
sample_img_file_path = os.path.join(sample_img_file_path, "PetImages")
sample_img_file_path = os.path.join(sample_img_file_path, "Cat")
sample_img_file_path = os.path.join(sample_img_file_path, "1.jpg")

image = cv.imread(sample_img_file_path, cv.IMREAD_GRAYSCALE)
cv.imwrite("1.jpg", image)





