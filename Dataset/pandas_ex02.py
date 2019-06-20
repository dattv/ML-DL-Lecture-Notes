import os

import numpy as np
import pandas

my_dict = {"name":["a", "b", "c", "d", "e", "f", "g"],
           "age":[20, 27, 35, 55, 18, 21, 35],
           "designation": ["VP", "CEO", "CFO", "VP", "VP", "CEO", "MD"]}

cf = pandas.DataFrame(my_dict)
print(cf)
root_path = os.path.dirname(os.path.dirname(__file__))

data_set_path = os.path.join(root_path, "Dataset")
full_file_path = os.path.join(data_set_path, "csv_data")
full_file_path = os.path.join(full_file_path, "pandas_ex02.csv")
cf.to_csv(full_file_path)

