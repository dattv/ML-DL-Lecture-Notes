import os

from tqdm import tqdm
from urllib.request import urlretrieve

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


def download(url=None, store_folder=None):
    if url == None:
        print("There are no ulr")

    else:
        tail = os.path.split(url)[1]
        file_name = tail.split(".")[0]
        if store_folder == None:
            store_folder = "../" + file_name

        if os.path.isdir(store_folder) == False:
            os.mkdir(store_folder)

        print("Downloading from: {}".format(url))
        with tqdm(unit="B", unit_scale=True, leave=True, miniters=1, desc=url.split("/")[-1]) as t:
            urlretrieve(url=url, filename=os.path.join(store_folder, file_name), reporthook=my_hook(t), data=None)

        print("Finish download")
