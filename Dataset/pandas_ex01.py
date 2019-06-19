import numpy as np
import pandas as pd
import os
import urllib.request
from tqdm import tqdm

def my_hook(t):
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner

dataset_url = "http://46.101.230.157/dilan/pandas_tutorial_read.csv"

root_path = os.path.dirname(os.path.dirname(__file__))

dataset_path = os.path.join(root_path, "Dataset")

full_data_path = os.path.join(dataset_path, "csv_data")
if os.path.isdir(full_data_path) == False:
    os.mkdir(full_data_path)

with tqdm(unit="B", unit_scale=True, leave=True, miniters=1, desc=dataset_url.split("/")[-1]) as t:
    urllib.request.urlretrieve(dataset_url, filename=full_data_path + "/pandas_tutorial_read.csv", reporthook=my_hook(t))

data = pd.read_csv(full_data_path, delimiter=";")

print(data)
