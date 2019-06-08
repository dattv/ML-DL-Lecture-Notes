import numpy
from PIL import Image

import os

file_path = os.getcwd()
file_path, _ = os.path.split(file_path)

file_path = os.path.join(file_path, "DeepLearning")
file_path = os.path.join(file_path, "kagglecatsanddogs_3367a")
file_path = os.path.join(file_path, "PetImages")
file_path = os.path.join(file_path, "Cat")

image_files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith(".jpg")]
image_files = numpy.asarray(image_files)

print(len(image_files))
print(image_files[0:10])

