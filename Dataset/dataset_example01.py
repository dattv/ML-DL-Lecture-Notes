# Create a random vector of shape (1000, 2)
import numpy as np
import tensorflow as tf

x = np.random.sample((1000, 2))
# Make a dataset from a numpy array
dataset = tf.data.Dataset.from_tensor_slices(x)

print(dataset)

# we can also pass more thatn one numpy array
features, labels = (np.random.sample((1000, 2)), np.random.sample((1000, 2)))
dataset = tf.data.Dataset.from_tensor_slices((features, labels))

