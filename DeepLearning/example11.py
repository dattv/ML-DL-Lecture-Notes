import os
import tarfile

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


class VGG:
    def __init__(self, VGG_URL="http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz",
                 ):
        self._vgg_url = VGG_URL
        self._vgg_file_name = self._vgg_url.split("/")[-1]
        temp_vgg_file_name = self._vgg_file_name
        temp_vgg_file_name = temp_vgg_file_name.split(".")[0]

        self._work_dir = os.getcwd()
        temp_vgg_file_name = os.path.join(self._work_dir, temp_vgg_file_name)
        if os.path.isdir(temp_vgg_file_name) == False:
            os.mkdir(temp_vgg_file_name)

        self._vgg_file_name = os.path.join(temp_vgg_file_name, self._vgg_file_name)

        # Download VGG model from the internet
        if not os.path.exists(self._vgg_file_name):
            with tqdm(unit="B", unit_scale=True, leave=True, miniters=1, desc=VGG_URL.split("/")[-1]) as t:
                self._vgg_file_name, _ = urllib.request.urlretrieve(self._vgg_url, filename=self._vgg_file_name,
                                                                    reporthook=my_hook(t), data=None)

        # Extract the the downloaded file
        if self._vgg_file_name.endswith("gz"):
            with tarfile.open(name=self._vgg_file_name) as tar:
                for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                    tar.extract(member=member, path=temp_vgg_file_name)



VGG16 = VGG()
