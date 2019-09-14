import tensorflow as tf
import tensorflow_estimator as TFE
from tensorflow._api.v1 import train


def VGG16_inference(input_tensor, mode, nb_class=1000, weight_decay=0.0005,
                    weight_init = tf.contrib.layers.xavier_initializer(uniform=False)):

    conv1 = tf.layers.conv2d(input_tensor, filters=64, kernel_size=3, strides=1, padding="same",
                             activation=tf.nn.relu, kernel_initializer=weight_init,
                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
    batch_norm1 = tf.layers.batch_normalization(conv1, training=mode == TFE.estimator.ModeKeys.TRAIN)
    drop_out1 = tf.layers.dropout(batch_norm1, rate=0.3, training=mode == TFE.estimator.ModeKeys.TRAIN)

    conv2 = tf.layers.conv2d(drop_out1, filters=64, kernel_size=3, strides=1, padding="same",
                             activation=tf.nn.relu, kernel_initializer=weight_init,
                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
    batch_norm2 = tf.layers.batch_normalization(conv2, training=mode == TFE.estimator.ModeKeys.TRAIN)

    max_pooling1 = tf.layers.max_pooling2d(batch_norm2, pool_size=2, strides=2)


    #--------
    conv3 = tf.layers.conv2d(max_pooling1, filters=128, kernel_size=3, strides=1, padding="same",
                             activation=tf.nn.relu, kernel_initializer=weight_init,
                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
    batch_norm3 = tf.layers.batch_normalization(conv3, training=mode == TFE.estimator.ModeKeys.TRAIN)
    drop_out3 = tf.layers.dropout(batch_norm3, rate=0.4, training=mode == TFE.estimator.ModeKeys.TRAIN)
    conv4 = tf.layers.conv2d(drop_out3, filters=128, kernel_size=3, strides=1, padding="same",
                             activation=tf.nn.relu, kernel_initializer=weight_init,
                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
    batch_norm4 = tf.layers.batch_normalization(conv4, training=mode == TFE.estimator.ModeKeys.TRAIN)
    max_pooling2 = tf.layers.max_pooling2d(batch_norm4, pool_size=2, strides=2)


    #--------
    conv5 = tf.layers.conv2d(max_pooling2, filters=256, kernel_size=3, strides=1, padding="same",
                             activation=tf.nn.relu, kernel_initializer=weight_init,
                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
    batch_norm5 = tf.layers.batch_normalization(conv5, training=mode == TFE.estimator.ModeKeys.TRAIN)
    drop_out4 = tf.layers.dropout(batch_norm5, rate=0.4, training=mode == TFE.estimator.ModeKeys.TRAIN)

    conv6 = tf.layers.conv2d(drop_out4, filters=256, kernel_size=3, strides=1, padding="same",
                             activation=tf.nn.relu, kernel_initializer=weight_init,
                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
    batch_norm6 = tf.layers.batch_normalization(conv6, training=mode == TFE.estimator.ModeKeys.TRAIN)
    drop_out5 = tf.layers.dropout(batch_norm6, rate=0.4, training=mode == TFE.estimator.ModeKeys.TRAIN)

    conv7 = tf.layers.conv2d(drop_out5, filters=256, kernel_size=3, strides=1, padding="same",
                             activation=tf.nn.relu, kernel_initializer=weight_init,
                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
    batch_norm7 = tf.layers.batch_normalization(conv7, training=mode == TFE.estimator.ModeKeys.TRAIN)

    max_pool3 = tf.layers.max_pooling2d(batch_norm7, pool_size=2, strides=2)

    #-------
    conv8 = tf.layers.conv2d(max_pool3, filters=512, kernel_size=3, strides=1, padding="same",
                             activation=tf.nn.relu, kernel_initializer=weight_init,
                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
    batch_norm8 = tf.layers.batch_normalization(conv8, training=mode == TFE.estimator.ModeKeys.TRAIN)
    drop_out6 = tf.layers.dropout(batch_norm8, rate=0.4, training=mode == TFE.estimator.ModeKeys.TRAIN)

    conv9 = tf.layers.conv2d(drop_out6, filters=512, kernel_size=3, strides=1, padding="same",
                             activation=tf.nn.relu, kernel_initializer=weight_init,
                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
    batch_norm9 = tf.layers.batch_normalization(conv9, training=mode == TFE.estimator.ModeKeys.TRAIN)
    drop_out7 = tf.layers.dropout(batch_norm9, rate=0.4, training=mode == TFE.estimator.ModeKeys.TRAIN)

    conv10 = tf.layers.conv2d(drop_out7, filters=512, kernel_size=3, strides=1, padding="same",
                             activation=tf.nn.relu, kernel_initializer=weight_init,
                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
    batch_norm10 = tf.layers.batch_normalization(conv10, training=mode == TFE.estimator.ModeKeys.TRAIN)

    max_pool4 = tf.layers.max_pooling2d(batch_norm10, pool_size=2, strides=2)

    #--------
    conv11 = tf.layers.conv2d(max_pool4, filters=512, kernel_size=3, strides=1, padding="same",
                             activation=tf.nn.relu, kernel_initializer=weight_init,
                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
    batch_norm11 = tf.layers.batch_normalization(conv11, training=mode == TFE.estimator.ModeKeys.TRAIN)
    drop_out8 = tf.layers.dropout(batch_norm11, rate=0.4, training=mode == TFE.estimator.ModeKeys.TRAIN)

    conv12 = tf.layers.conv2d(drop_out8, filters=512, kernel_size=3, strides=1, padding="same",
                              activation=tf.nn.relu, kernel_initializer=weight_init,
                              kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
    batch_norm12 = tf.layers.batch_normalization(conv12, training=mode == TFE.estimator.ModeKeys.TRAIN)
    drop_out9 = tf.layers.dropout(batch_norm12, rate=0.4, training=mode == TFE.estimator.ModeKeys.TRAIN)

    conv13 = tf.layers.conv2d(drop_out9, filters=512, kernel_size=3, strides=1, padding="same",
                              activation=tf.nn.relu, kernel_initializer=weight_init,
                              kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
    batch_norm13 = tf.layers.batch_normalization(conv13, training=mode == TFE.estimator.ModeKeys.TRAIN)

    max_pool5 = tf.layers.max_pooling2d(batch_norm13, pool_size=2, strides=2)

    drop_out10 = tf.layers.dropout(max_pool5, rate=0.5, training=mode == TFE.estimator.ModeKeys.TRAIN)

    #---------
    flatten = tf.layers.flatten(drop_out10)
    fully1 = tf.layers.dense(flatten, 512, activation=tf.nn.relu)
    batch_norm14 = tf.layers.batch_normalization(fully1, training=mode == TFE.estimator.ModeKeys.TRAIN)
    drop_out11 = tf.layers.dropout(batch_norm14, rate=0.5, training=mode == TFE.estimator.ModeKeys.TRAIN)
    output = tf.layers.dense(drop_out11, nb_class, activation=None)

    return output












