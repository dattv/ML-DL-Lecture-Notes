import tensorflow as tf
import numpy as np

def fully_connected_layer(input_layer, units):
    return tf.layers.dense(input_layer,
                           units=units,
                           activation=tf.nn.relu)

def convolution_layer(input_layer, filter_size):
    return tf.layers.conv2d(input_layer,
                            filters=filter_size,
                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            kernel_size=3,
                            strides=2)

def deconvolution_layer(input_layer, filter_size, activation=tf.nn.relu):
    return tf.layers.conv2d_transpose(input_layer,
                                      filters=filter_size,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                      kernel_size=3,
                                      activation=activation,
                                      strides=2)

# Define the converging encoder with five layers of convolution, as shown in the following code:
input_layer = tf.placeholder(tf.float32, [None, 128, 128, 3])
convolution_layer_1 = convolution_layer(input_layer, 1024)
convolution_layer_2 = convolution_layer(convolution_layer_1, 512)
convolution_layer_3 = convolution_layer(convolution_layer_2, 256)
convolution_layer_4 = convolution_layer(convolution_layer_3, 128)
convolution_layer_5 = convolution_layer(convolution_layer_4, 32)

#
convolution_layer_5_flattened = tf.layers.flatten(convolution_layer_5)
bottleneck_layer = fully_connected_layer(convolution_layer_5_flattened, 16)
c5_shape = convolution_layer_5.get_shape().as_list()
c5f_flat_shape = convolution_layer_5_flattened.get_shape().as_list()[1]
fully_connected = fully_connected_layer(bottleneck_layer,
                                        c5f_flat_shape)
fully_connected = tf.reshape(fully_connected,
                             [-1, c5f_flat_shape])

