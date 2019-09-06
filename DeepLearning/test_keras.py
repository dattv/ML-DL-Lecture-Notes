import tensorflow as tf
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

TensorBoard = tf.keras.callbacks.TensorBoard

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = np.reshape(train_images, [-1, 28, 28, 1])
test_images = np.reshape(test_images, [-1, 28, 28, 1])

train_images = train_images[0:100]
test_images = test_images[0:100]

train_labels = train_labels[0:100]
test_labels = test_labels[0:100]

keras = tf.keras
Input = tf.keras.Input

weight_init = tf.contrib.layers.xavier_initializer(uniform=False)
weight_decay = 0.0005
bias = False
channel_axis = -1

k = 8
depth = 16
n_stages = [16, 16 * k, 32 * k, 64 * k]
n = (depth - 4) / 6
dropout_probability =0.9


def _wide_basic(n_input_plane, n_output_plane, stride):

    def f(net):
        # format of conv_params:
        #               [ [kernel_size=("kernel width", "kernel height"),
        #               strides="(stride_vertical,stride_horizontal)",
        #               padding="same" or "valid"] ]
        # B(3,3): orignal <<basic>> block
        conv_params = [[3, stride, "same"],
                       [3, 1, "same"]]

        n_bottleneck_plane = n_output_plane

        # Residual block
        for i, v in enumerate(conv_params):

            if i == 0:
                # if n_input_plane != n_output_plane:
                    # net = tf.keras.layers.BatchNormalization(axis=-1, fused=False)(net)
                #     net = tf.keras.layers.Activation("relu")(net)
                #     convs = net
                # else:
                #     convs = tf.keras.layers.BatchNormalization(axis=-1, fused=False)(net)
                #     convs = tf.keras.layers.Activation("relu")(convs)

                convs = tf.keras.layers.Conv2D(n_bottleneck_plane, kernel_size=v[0],
                                               strides=v[1],
                                               padding=v[2],
                                               use_bias=False,
                                               kernel_initializer=weight_init,
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                               activation=None)(net)
                convs = tf.keras.layers.BatchNormalization(fused=False)(convs)
                # convs = tf.keras.layers.Activation("relu")(convs)

                convs = tf.keras.layers.Conv2D(n_bottleneck_plane, kernel_size=v[0],
                                               strides=v[1],
                                               padding=v[2],
                                               use_bias=False,
                                               kernel_initializer=weight_init,
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                               activation="relu")(
                    convs)
                convs = tf.keras.layers.BatchNormalization(fused=False)(convs)
                # if n_input_plane != n_output_plane:
                # convs = tf.keras.layers.BatchNormalization(fused=False)(conv1)
            else:
                # convs = tf.keras.layers.BatchNormalization(axis=-1, fused=False)(convs)
                # convs = tf.keras.layers.Activation("relu")(convs)
                # convs = tf.nn.relu(convs)
                # if dropout_probability > 0:
                #     convs = tf.keras.layers.Dropout(dropout_probability)(convs)
                convs = tf.keras.layers.Conv2D(n_bottleneck_plane, kernel_size=v[0],
                                               strides=v[1],
                                               padding=v[2],
                                               use_bias=False,
                                               kernel_initializer=weight_init,
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                               )(convs)


        # Shortcut Connection: identity function or 1x1 convolutional
        #  (depends on difference between input & output shape - this
        #   corresponds to whether we are using the first block in each
        #   group; see _layer() ).
        # if n_input_plane != n_output_plane:
        #     shortcut = tf.keras.layers.Conv2D(n_output_plane, kernel_size=1,
        #                                       strides=stride,
        #                                       padding="same",
        #                                       kernel_initializer=weight_init,
        #                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
        #                                       use_bias=False)(net)
        # else:
        #     shortcut = net
        #
        # return tf.keras.layers.add([convs, shortcut])
        return convs

    return f

def _layer( block, n_input_plane, n_output_plane, count, stride):
    def f(net):
        net = block(n_input_plane, n_output_plane, stride)(net)
        for i in range(2, int(count + 1)):
            net = block(n_output_plane, n_output_plane, stride=1)(net)
        return net

    return f


def build_keras_model():
    inputs = Input(shape=(28, 28, 1))
    #

    conv2 = tf.keras.layers.Conv2D(16, kernel_size=3, strides=1, activation="relu", padding="same", use_bias=False)(inputs)
    batch_norm2 = tf.keras.layers.BatchNormalization(fused=False)(conv2)
    conv2 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, activation="relu", padding="same", use_bias=False)(batch_norm2)
    conv2 = tf.keras.layers.BatchNormalization(fused=False)(conv2)
    conv2 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, activation="relu", padding="same", use_bias=False)(conv2)

    conv = tf.keras.layers.Conv2D(16, kernel_size=3, strides=1, activation="relu", padding="same", use_bias=False)(inputs)
    conv = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, activation="relu", padding="same", use_bias=False)(conv)
    conv = tf.keras.layers.BatchNormalization(fused=False)(conv)
    conv = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, activation="relu", padding="same", use_bias=False)(conv)

    conv = tf.keras.layers.add([conv2, conv])
    conv = tf.keras.layers.BatchNormalization(fused=False)(conv)

    conv = tf.keras.layers.AveragePooling2D(pool_size=7)(conv)
    conv = tf.keras.layers.Flatten()(conv)

    conv2 = tf.keras.layers.Dense(10, activation='softmax')(conv)

    conv = tf.keras.layers.Dense(10, activation='softmax')(conv)

    model = tf.keras.Model(inputs=inputs, outputs=[conv, conv2])
    return model


# train
train_graph = tf.Graph()
train_sess = tf.Session(graph=train_graph)

keras.backend.set_session(train_sess)
with train_graph.as_default():
    train_model = build_keras_model()

    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                              write_graph=True, write_images=False, update_freq="batch")

    tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=100)
    train_sess.run(tf.global_variables_initializer())

    train_model.compile(
        optimizer='adam',
        loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy'],
        metrics=['accuracy'],
    )

    train_model.fit(train_images[0:100], [train_labels[0:100], train_labels[0:100]], epochs=1, callbacks=[tensorboard])

    # save graph and checkpoints
    saver = tf.train.Saver()
    saver.save(train_sess, 'checkpoint/checkpoint.ckpt')

with train_graph.as_default():
    print('sample result of original model')
    print(train_model.predict(test_images[:1]))


# eval
eval_graph = tf.Graph()
eval_sess = tf.Session(graph=eval_graph)

keras.backend.set_session(eval_sess)

with eval_graph.as_default():
    keras.backend.set_learning_phase(0)
    eval_model = build_keras_model()

    tf.contrib.quantize.create_eval_graph(input_graph=eval_graph)
    eval_graph_def = eval_graph.as_graph_def()
    saver = tf.train.Saver()
    saver.restore(eval_sess, 'checkpoint/checkpoint.ckpt')

    temp = eval_model.output
    name = []
    for t in temp:
        name.append(t.op.name)

    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        eval_sess,
        eval_graph_def,
        name
    )

    with open('frozen_model.pb', 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())