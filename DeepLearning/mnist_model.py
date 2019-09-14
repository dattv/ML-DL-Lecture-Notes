import tensorflow as tf
import tensorflow_estimator as TFE


def mnist_inference(input_tensor, mode, nb_class=100):
    conv1 = tf.layers.conv2d(inputs=input_tensor, filters=64, kernel_size=[3, 3], padding="same",
                             activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

    conv3_shape = conv3.shape
    pool2_flat = tf.reshape(conv3, [-1, int(conv3_shape[1] * conv3_shape[2] * conv3_shape[3])])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == TFE.estimator.ModeKeys.TRAIN)

    output = tf.layers.dense(inputs=dropout, units=nb_class)

    return output

