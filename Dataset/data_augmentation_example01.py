import os
import urllib.request

import numpy
from scipy import ndimage

from matplotlib import pyplot as plt
from PIL import Image

root_path = os.path.dirname(os.path.dirname(__file__))
dataset_path = os.path.join(root_path, "Dataset")
temp = __file__
temp = temp.split("/")
temp = temp[len(temp) - 1]
temp = temp.split(".")[0]
dataset_path = os.path.join(dataset_path, temp)
if os.path.isdir(dataset_path) == False:
    os.mkdir(dataset_path)

FeatheredFriend_url = "https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/01/bird.jpg"

file_name = FeatheredFriend_url.split("/")
file_name = file_name[len(file_name) - 1]
file_name = os.path.join(dataset_path, file_name)

urllib.request.urlretrieve(FeatheredFriend_url, filename=file_name)

# start to augmentation data

img = Image.open(file_name)
img = numpy.asarray(img)

# Flipping
flippedlr_img = numpy.fliplr(img)

flippedud_img = numpy.flipud(img)

class Augmentation_image():
    def __init__(self, file_name=None):
        print("data augmentation")
        self._img= Image.open(file_name)

        self._img_numpy = numpy.asarray(self._img)

        self._file_name_prefix = file_name.split("/")
        self._file_name_prefix = self._file_name_prefix[len(self._file_name_prefix) - 2]
        self._file_name_prefix = os.path.join(self._file_name_prefix, "augmentation")

        if os.path.isdir(self._file_name_prefix) == False:
            os.mkdir(self._file_name_prefix)

        temp = file_name.split(".")[0]
        temp = temp.split("/")
        temp = temp[len(temp) - 1]
        self._file_name_prefix = os.path.join(self._file_name_prefix, temp)

        self._file_name_suffixes = file_name.split(".")[1]

        # flip original image
        self.flip_img(self._img_numpy)

        self.blur_img(self._img_numpy)

        self.rotate_img(self._img_numpy)

    def flip_img(self, numpy_image=None):
        if numpy_image is not None:
            img_flipped = numpy.fliplr(numpy_image)
            img = Image.fromarray(img_flipped)
            img.save(self._file_name_prefix + "_lr." + self._file_name_suffixes)

            img_flipped = numpy.flipud(img_flipped)
            img = Image.fromarray(img_flipped)
            img.save(self._file_name_prefix + "_lr_ud." + self._file_name_suffixes)

            img_flipped = numpy.flipud(numpy_image)
            img = Image.fromarray(img_flipped)
            img.save(self._file_name_prefix + "_ud." + self._file_name_suffixes)

    def rotate_img(self, numpy_image=None):
        if numpy_image is not None:
            for i in range (1, 360, 20):
                img = Image.fromarray(numpy_image)
                rot_img = img.rotate(i)
                rot_img.save(self._file_name_prefix + "_rotate_" + str(i) + "." + self._file_name_suffixes)

    def blur_img(self, numpy_image=None):
        if numpy_image is not None:
            alpha = 30
            print(numpy_image.shape)
            for i in range(1, 5):
                blurred_f = ndimage.gaussian_filter(numpy_image, sigma=i/2.)
                filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
                sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)

                img = Image.fromarray(blurred_f)
                img.save(self._file_name_prefix + "_gaussian_" + str(i) + "." + self._file_name_suffixes)

                # img_sharpened = Image.fromarray(sharpened)
                # img_sharpened.save(self._file_name_prefix + "_sharpened_" + str(i) + "." + self._file_name_suffixes)


def main ():
    Augmentation_image(file_name)


if __name__ == '__main__':
    main()




