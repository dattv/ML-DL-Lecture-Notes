from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
from pathlib import Path
import tensorflow as tf

print(tf.__version__)

# Setup logging
Path('results').mkdir(exist_ok=True)
tf.logging.set_verbosity(logging.INFO)
handlers = [
    logging.FileHandler('results/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers


def generator_fn():
    for digit in range(2):
        line = 'I am digit {}'.format(digit)
        words = line.split()
        yield [w.encode() for w in words], len(words)


for words in generator_fn():
    print(words)

shapes = ([None], ())
types = (tf.string, tf.int32)

dataset = tf.data.Dataset.from_generator(generator_fn, output_shapes=shapes, output_types=types)


iterator = dataset.make_one_shot_iterator()
node = iterator.get_next()
with tf.Session(graph=tf.get_default_graph()) as sess:
    print(sess.run(node))
    print(sess.run(node))