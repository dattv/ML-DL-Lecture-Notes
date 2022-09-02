
import sys
import os

import numpy as np
import tensorflow as tf
import random

SEED = 12345
def set_seeds(seed):
    """

    :param seed:
    :return:
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed=SEED):
    """

    :param seed:
    :return:
    """
    set_seeds(seed=seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
