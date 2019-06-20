import pandas as pd
import numpy as np
import os

import urllib.request
from tqdm import tqdm

zoo_url = "http://46.101.230.157/datacoding101/zoo.csv"

def my_hook(t):
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner

root_path = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(root_path, "Dataset")
data_path = os.path.join(data_path, "csv_data")

full_file_path = data_path + "/zoo.csv"
with tqdm(unit="B", unit_scale=True, leave=True, miniters=1, desc=zoo_url.split("/")[-1]) as t:
    urllib.request.urlretrieve(zoo_url, filename=full_file_path, reporthook=my_hook(t))

zoo_data = pd.read_csv(full_file_path, delimiter=",")
print(zoo_data)

# count the number of rows
print(zoo_data.count())

#
print(zoo_data[['animal']].count())

#
print(zoo_data.animal.count())

# pandas data aggregation
print(zoo_data.water_need.sum())

#
print(zoo_data.sum())

#
print(zoo_data.water_need.min())
print(zoo_data.water_need.max())

#
print(zoo_data.water_need.mean())
print(zoo_data.water_need.median())

# grouping in pandas
animal = zoo_data.groupby("animal")
print(animal.mean())