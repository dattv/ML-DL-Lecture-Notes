import cv2 as cv
import numpy as np
import os

root_path = os.path.dirname(os.path.dirname(__file__))
sample_img_file_path = os.path.join(root_path, "DeepLearning")
sample_img_file_path = os.path.join(sample_img_file_path, "kagglecatsanddogs_3367a")
sample_img_file_path = os.path.join(sample_img_file_path, "PetImages")
sample_img_file_path = os.path.join(sample_img_file_path, "Cat")
sample_img_file_path = os.path.join(sample_img_file_path, "1.jpg")

image = cv.imread(sample_img_file_path)
image[0, 0] = [255, 255, 255]
cv.imshow("image", image)
cv.waitKey(0)
cv.destroyAllWindows()


print(image.item(20, 20, 0))
image.itemset((20, 20, 0), 12)
print(image.item(20, 20, 0))

# setting all green value to 0
image[:, :, 1] = 0
cv.imshow("green", image)
cv.waitKey(0)
cv.destroyAllWindows()

# copy and roi
my_roi = image[0:100, 0:100]
image[100:200, 100:200] = my_roi
cv.imshow("roi", image)
cv.waitKey(0)
cv.destroyAllWindows()

print(image.shape)
print(image.size)
print(image.dtype)

