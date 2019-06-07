import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices(tf.random_uniform([1000, 2]))
