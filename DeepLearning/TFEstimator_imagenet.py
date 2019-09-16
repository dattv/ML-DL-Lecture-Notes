import shutil

import tensorflow as tf
import tensorflow_estimator as TFE
import os
import random
import time
import numpy as np
from DeepLearning.cifar10_model import cifar10_inference

imageWidth = 224
imageHeight = 224
imageDepth = 3
batch_size = 32
resize_min = 256

train_files_names = os.listdir("/media/dat/68fa98f8-9d03-4c1e-9bdb-c71ea72ab6fa/dat/dataset/tfrecord/tfrecord_ImageNet/")
train_files = ['/media/dat/68fa98f8-9d03-4c1e-9bdb-c71ea72ab6fa/dat/dataset/tfrecord/tfrecord_ImageNet/' + item for item in train_files_names]
valid_files_names = os.listdir('/media/dat/68fa98f8-9d03-4c1e-9bdb-c71ea72ab6fa/dat/dataset/tfrecord/tfrecord_ImageNet/')
valid_files = ['/media/dat/68fa98f8-9d03-4c1e-9bdb-c71ea72ab6fa/dat/dataset/tfrecord/tfrecord_ImageNet/' + item for item in valid_files_names]


# Parse TFRECORD and distort the image for train
def _parse_function(example_proto):
    features = {"image/encoded": tf.FixedLenFeature([], tf.string, default_value=""),
                "image/height": tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                "image/width": tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                "image/channels": tf.FixedLenFeature([1], tf.int64, default_value=[3]),
                "image/colorspace": tf.FixedLenFeature([], tf.string, default_value=""),
                "image/format": tf.FixedLenFeature([], tf.string, default_value=""),
                "image/class/label": tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                "image/object/bbox/xmin": tf.VarLenFeature(tf.float32),
                "image/object/bbox/xmax": tf.VarLenFeature(tf.float32),
                "image/object/bbox/ymin": tf.VarLenFeature(tf.float32),
                "image/object/bbox/ymax": tf.VarLenFeature(tf.float32),
                "image/class/text": tf.FixedLenFeature([], tf.string, default_value=""),
                "image/filename": tf.FixedLenFeature([], tf.string, default_value="")
                }
    parsed_features = tf.parse_single_example(example_proto, features)
    image_decoded = tf.image.decode_jpeg(parsed_features["image/encoded"], channels=3)
    # Random resize the image
    shape = tf.shape(image_decoded)
    height, width = shape[0], shape[1]
    resized_height, resized_width = tf.cond(height < width,
                                            lambda: (resize_min, tf.cast(
                                                tf.multiply(tf.cast(width, tf.float64), tf.divide(resize_min, height)),
                                                tf.int32)),
                                            lambda: (tf.cast(
                                                tf.multiply(tf.cast(height, tf.float64), tf.divide(resize_min, width)),
                                                tf.int32), resize_min))
    image_float = tf.image.convert_image_dtype(image_decoded, tf.float32)
    resized = tf.image.resize_images(image_float, [resized_height, resized_width])
    # Random crop from the resized image
    cropped = tf.random_crop(resized, [imageHeight, imageWidth, 3])
    # Flip to add a little more random distortion in.
    flipped = tf.image.random_flip_left_right(cropped)
    # Standardization the image
    # image_train = flipped
    image_train = tf.image.per_image_standardization(flipped)
    features = {'images': image_train}
    # return image_train, tf.one_hot(parsed_features["label"][0], 1000)
    return features, parsed_features["image/class/label"][0]


def train_input_fn():
    dataset_train = tf.data.TFRecordDataset(train_files)
    dataset_train = dataset_train.map(_parse_function, num_parallel_calls=4)
    dataset_train = dataset_train.repeat(10)
    dataset_train = dataset_train.batch(batch_size)
    dataset_train = dataset_train.prefetch(batch_size)
    return dataset_train


def _parse_test_function(example_proto):
    features = {"image/encoded": tf.FixedLenFeature([], tf.string, default_value=""),
                "image/height": tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                "image/width": tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                "image/channels": tf.FixedLenFeature([1], tf.int64, default_value=[3]),
                "image/colorspace": tf.FixedLenFeature([], tf.string, default_value=""),
                "image/format": tf.FixedLenFeature([], tf.string, default_value=""),
                "image/class/label": tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                "image/object/bbox/xmin": tf.VarLenFeature(tf.float32),
                "image/object/bbox/xmax": tf.VarLenFeature(tf.float32),
                "image/object/bbox/ymin": tf.VarLenFeature(tf.float32),
                "image/object/bbox/ymax": tf.VarLenFeature(tf.float32),
                "image/class/text": tf.FixedLenFeature([], tf.string, default_value=""),
                "image/filename": tf.FixedLenFeature([], tf.string, default_value="")
                }
    parsed_features = tf.parse_single_example(example_proto, features)
    image_decoded = tf.image.decode_jpeg(parsed_features["image/encoded"], channels=3)
    shape = tf.shape(image_decoded)
    height, width = shape[0], shape[1]
    resized_height, resized_width = tf.cond(height < width,
                                            lambda: (resize_min, tf.cast(
                                                tf.multiply(tf.cast(width, tf.float64), tf.divide(resize_min, height)),
                                                tf.int32)),
                                            lambda: (tf.cast(
                                                tf.multiply(tf.cast(height, tf.float64), tf.divide(resize_min, width)),
                                                tf.int32), resize_min))
    image_float = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_resized = tf.image.resize_images(image_float, [resized_height, resized_width])

    # calculate how many to be center crop
    shape = tf.shape(image_resized)
    height, width = shape[0], shape[1]
    amount_to_be_cropped_h = (height - imageHeight)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - imageWidth)
    crop_left = amount_to_be_cropped_w // 2
    image_cropped = tf.slice(image_resized, [crop_top, crop_left, 0], [imageHeight, imageWidth, -1])
    image_valid = tf.image.per_image_standardization(image_cropped)
    features = {'images': image_valid}
    # return image_valid, tf.one_hot(parsed_features["label"][0], 1000)
    return features, parsed_features["image/class/label"][0]


def val_input_fn():
    dataset_valid = tf.data.TFRecordDataset(valid_files)
    dataset_valid = dataset_valid.map(_parse_test_function, num_parallel_calls=4)
    dataset_valid = dataset_valid.batch(batch_size)
    dataset_valid = dataset_valid.prefetch(batch_size)
    return dataset_valid


def predict_input_fn():
    dataset_valid = tf.data.TFRecordDataset(valid_files)
    dataset_valid = dataset_valid.map(_parse_test_function, num_parallel_calls=4)
    dataset_valid = dataset_valid.take(batch_size)
    dataset_valid = dataset_valid.batch(batch_size)
    return dataset_valid


l = tf.keras.layers


def _conv(inputs, filters, kernel_size, strides, padding, bias=False, normalize=True, activation='relu'):
    output = inputs
    padding_str = 'same'
    if padding > 0:
        output = l.ZeroPadding2D(padding=padding)(output)
        padding_str = 'valid'
    output = l.Conv2D(filters, kernel_size, strides, padding_str, use_bias=bias,
                      kernel_initializer='he_normal',
                      kernel_regularizer=tf.keras.regularizers.l2(l=5e-4))(output)
    if normalize:
        output = l.BatchNormalization(axis=3)(output)
    if activation == 'relu':
        output = l.ReLU()(output)
    if activation == 'relu6':
        output = l.ReLU(max_value=6)(output)
    if activation == 'leaky_relu':
        output = l.LeakyReLU(alpha=0.1)(output)
    return output


def _dwconv(inputs, filters, kernel_size, strides, padding, bias=False, activation='relu'):
    output = inputs
    padding_str = 'same'
    if padding > 0:
        output = l.ZeroPadding2D(padding=(padding, padding))(output)
        padding_str = 'valid'
    output = l.DepthwiseConv2D(kernel_size, strides, padding_str, use_bias=bias,
                               depthwise_initializer='he_uniform',
                               depthwise_regularizer=tf.keras.regularizers.l2(l=5e-4))(output)
    output = l.BatchNormalization(axis=3)(output)
    if activation == 'relu':
        output = l.ReLU()(output)
    if activation == 'relu6':
        output = l.ReLU(max_value=6)(output)
    if activation == 'leaky_relu':
        output = l.LeakyReLU(alpha=0.1)(output)
    return output


def _bottleneck(inputs, in_filters, out_filters, kernel_size, strides, bias=False, activation='relu6', t=1):
    output = inputs
    output = _conv(output, in_filters * t, 1, 1, 0, False, activation)
    padding = 0
    if strides == 2:
        padding = 1
    output = _dwconv(output, in_filters * t, kernel_size, strides, padding, bias=False, activation=activation)
    output = _conv(output, out_filters, 1, 1, 0, False, 'linear')
    if strides == 1 and inputs.get_shape().as_list()[3] == output.get_shape().as_list()[3]:
        output = l.add([output, inputs])
    return output


def mobilenet_model_v1():
    # Input Layer
    image = tf.keras.Input(shape=(imageHeight, imageWidth, 3))
    net = _conv(image, 32, 3, 2, 1)
    net = _dwconv(net, 32, 3, 1, 0)
    net = _conv(net, 64, 1, 1, 0)
    net = _dwconv(net, 64, 3, 2, 1)
    net = _conv(net, 128, 1, 1, 0)
    net = _dwconv(net, 128, 3, 1, 0)
    net = _conv(net, 128, 1, 1, 0)
    net = _dwconv(net, 128, 3, 2, 1)
    net = _conv(net, 256, 1, 1, 0)
    net = _dwconv(net, 256, 3, 1, 0)
    net = _conv(net, 256, 1, 1, 0)
    net = _dwconv(net, 256, 3, 2, 1)
    net = _conv(net, 512, 1, 1, 0)
    for _ in range(5):
        net = _dwconv(net, 512, 3, 1, 0)
        net = _conv(net, 512, 1, 1, 0)
    net = _dwconv(net, 512, 3, 2, 1)
    net = _conv(net, 1024, 1, 1, 0)
    net = _dwconv(net, 1024, 3, 1, 0)
    net = _conv(net, 1024, 1, 1, 0)
    net = l.GlobalAveragePooling2D()(net)
    net = l.Flatten()(net)
    logits = l.Dense(1000, kernel_initializer=tf.initializers.truncated_normal(stddev=1 / 1000))(net)
    model = tf.keras.Model(inputs=image, outputs=logits)
    return model


def mobilenet_model_v2():
    # Input Layer
    image = tf.keras.Input(shape=(imageHeight, imageWidth, 3))  # 224*224*3
    net = _conv(image, 32, 3, 2, 1, False, 'relu6')  # 112*112*32
    net = _bottleneck(net, 32, 16, 3, 1, False, 'relu6', 1)  # 112*112*16
    net = _bottleneck(net, 16, 24, 3, 2, False, 'relu6', 6)  # 56*56*24
    # net = _bottleneck(net, 24, 24, 3, 1, False, 'relu6', 6)  # 56*56*24
    net = _bottleneck(net, 24, 32, 3, 2, False, 'relu6', 6)  # 28*28*32
    # net = _bottleneck(net, 32, 32, 3, 1, False, 'relu6', 6)  # 28*28*32
    # net = _bottleneck(net, 32, 32, 3, 1, False, 'relu6', 6)  # 28*28*32
    net = _bottleneck(net, 32, 64, 3, 2, False, 'relu6', 6)  # 14*14*64
    # net = _bottleneck(net, 64, 64, 3, 1, False, 'relu6', 6)  # 14*14*64
    # net = _bottleneck(net, 64, 64, 3, 1, False, 'relu6', 6)  # 14*14*64
    # net = _bottleneck(net, 64, 64, 3, 1, False, 'relu6', 6)  # 14*14*64
    net = _bottleneck(net, 64, 96, 3, 1, False, 'relu6', 6)  # 14*14*96
    # net = _bottleneck(net, 96, 96, 3, 1, False, 'relu6', 6)  # 14*14*96
    # net = _bottleneck(net, 96, 96, 3, 1, False, 'relu6', 6)  # 14*14*96
    # net = _bottleneck(net, 96, 96, 3, 1, False, 'relu6', 6)  # 14*14*96
    net = _bottleneck(net, 96, 160, 3, 2, False, 'relu6', 6)  # 7*7*160
    # net = _bottleneck(net, 160, 160, 3, 1, False, 'relu6', 6)  # 7*7*160
    # net = _bottleneck(net, 160, 160, 3, 1, False, 'relu6', 6)  # 7*7*160
    # net = _bottleneck(net, 160, 320, 3, 1, False, 'relu6', 6)  # 7*7*320
    net = _conv(net, 1280, 3, 1, 0, False, 'relu6')  # 7*7*1280
    net = l.AveragePooling2D(7)(net)
    net = l.Flatten()(net)
    logits = l.Dense(1000, kernel_initializer=tf.initializers.truncated_normal(stddev=1 / 1000))(net)
    model = tf.keras.Model(inputs=image, outputs=logits)
    return model


def mobilenet(features, labels, mode, params):
    model = mobilenet_model_v2()
    training = (mode == TFE.estimator.ModeKeys.TRAIN)
    # features = tf.reshape(features, [-1,imageHeight,imageWidth,3])
    images = tf.reshape(features["images"], [-1, imageHeight, imageWidth, 3])
    logits = model(images, training)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=-1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == TFE.estimator.ModeKeys.PREDICT:
        return TFE.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                                          export_outputs={'classify': TFE.estimator.export.PredictOutput(predictions)})

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == TFE.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        boundaries = [5000, 60000, 80000]
        values = [0.1, 0.01, 0.001, 0.0001]
        learning_rate = 1.e-5#tf.compat.v1.train.piecewise_constant(global_step, boundaries, values)
        tf.summary.scalar('learning_rate', learning_rate)
        # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        optimizer = tf.train.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(model.get_updates_for(features)):
            train_op = optimizer.minimize(loss=loss, global_step=global_step)
        return TFE.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    m = tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true=labels, y_pred=logits)
    tf.summary.scalar('top-5_accuracy', m)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    tf.summary.scalar('accuracy', accuracy[0])
    eval_metric_ops = {
        # "accuracy": tf.metrics.accuracy(labels=true_labels, predictions=predictions["classes"])}
        "accuracy": accuracy}
    # "top-5 accuracy": (m.result(), m.update_state(y_true=labels, y_pred=logits))}
    return TFE.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(_):
    print("Removing previous artifacts...")
    shutil.rmtree("./ImageNet_model/", ignore_errors=True)
    my_feature_columns = []
    my_feature_columns.append(tf.feature_column.numeric_column(key='images', shape=(imageHeight, imageWidth, 3)))
    imagenet_classifier = TFE.estimator.Estimator(model_fn=mobilenet,
                                                 model_dir="./ImageNet_model/",
                                                 params={'feature_columns': my_feature_columns, })
    for _ in range(10):
        imagenet_classifier.train(input_fn=train_input_fn, steps=5000)
        eval_results = imagenet_classifier.evaluate(input_fn=val_input_fn)
        print(eval_results)


if __name__ == "__main__":
    tf.app.run(main)
