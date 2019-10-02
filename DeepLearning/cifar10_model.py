import tensorflow as tf
import tensorflow_estimator as TFE

def cifar10_inference(images, mode, NUM_CLASSES):
    conv1 = tf.layers.conv2d(images, filters=64, kernel_size=3, strides=(1, 1),
                             padding="same", activation=tf.nn.relu)

    pooling1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

    batch_norm1 = tf.layers.batch_normalization(pooling1, training=mode)

    conv2 = tf.layers.conv2d(batch_norm1, filters=128, kernel_size=3, strides=(1, 1),
                             padding="same", activation=tf.nn.relu)

    pooling2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)

    batch_norm2 = tf.layers.batch_normalization(pooling2, training=mode)

    conv3 = tf.layers.conv2d(batch_norm2, filters=256, kernel_size=5, strides=(1, 1),
                             padding="same", activation=tf.nn.relu)

    pooling3 = tf.layers.max_pooling2d(conv3, pool_size=2, strides=2)

    batch_norm3 = tf.layers.batch_normalization(pooling3, training=mode)

    conv4 = tf.layers.conv2d(batch_norm3, filters=512, kernel_size=5, strides=(1, 1),
                             padding="same", activation=tf.nn.relu)

    pooling4 = tf.layers.max_pooling2d(conv4, pool_size=2, strides=2)

    batch_norm4 = tf.layers.batch_normalization(pooling4, training=mode)

    flat = tf.layers.flatten(batch_norm4)

    full1 = tf.layers.dense(flat, 128, activation=tf.nn.relu)
    full1 = tf.layers.dropout(full1, rate=0.7, training=mode)
    full1 = tf.layers.batch_normalization(full1, training=mode)

    full2 = tf.layers.dense(full1, 56, activation=tf.nn.relu)
    full2 = tf.layers.dropout(full2, rate=0.7, training=mode)
    full2 = tf.layers.batch_normalization(full2, training=mode)

    full3 = tf.layers.dense(full2, 512, activation=tf.nn.relu)
    full3 = tf.layers.dropout(full3, rate=0.7, training=mode)
    full3 = tf.layers.batch_normalization(full3, training=mode)

    full4 = tf.layers.dense(full3, 1024, activation=tf.nn.relu)
    full4 = tf.layers.dropout(full4, rate=0.7, training=mode)
    full4 = tf.layers.batch_normalization(full4, training=mode)

    out = tf.layers.dense(full4, NUM_CLASSES, activation=None)

    return out
