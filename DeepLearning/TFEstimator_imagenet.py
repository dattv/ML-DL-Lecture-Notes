import shutil

import cv2
import tensorflow as tf
import tensorflow_estimator as TFE
import os
import random
import time
import numpy as np
from pathlib import Path
from DeepLearning.cifar10_model import cifar10_inference
from DeepLearning.MobileNet import *
from tensorflow.python.client import device_lib

tf.logging.set_verbosity(tf.logging.INFO)

IMAGEWIDTH = 64
IMAGEHEIGHT = IMAGEWIDTH
IMAGEDEPTH = 3
BATCH_SIZE = 32
RESIZE_MIN = 64
NB_EPOCH = 60
NB_CLASSES = 1000
RUN_FROM_LAST = True
NB_TRAIN_IMG = 15360#1281167
NB_VALID_IMG = 15360#50000
NB_TEST_IMG = 150000
SAVE_CHECKPOINT_STEP = NB_TRAIN_IMG // BATCH_SIZE
TF_RANDOM_SEED = 19851211
file_name = __file__
file_name = os.path.split(file_name)[1]
file_name = file_name.split(".")[0]
model_file = "./" + file_name + "_model"
Path(model_file).mkdir(exist_ok=True)
MODEL_NAME = model_file

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

name_gpus = get_available_gpus()
if len(name_gpus) > 1:
    NUM_GPUS = len(name_gpus) - 1
else:
    NUM_GPUS = len(name_gpus)

print("USE: {} GPUS FOR TRAINING".format(NUM_GPUS))
train_dir = "/media/dat/68fa98f8-9d03-4c1e-9bdb-c71ea72ab6fa/dat/dataset/tfrecord/tfrecord_ImageNet"
valid_dir = "/media/dat/68fa98f8-9d03-4c1e-9bdb-c71ea72ab6fa/dat/dataset/tfrecord/tfrecord_ImageNet"

train_files_names = os.listdir(train_dir)
train_files = [os.path.join(train_dir, item) for item in train_files_names
               if item.split("-")[0] == "train"]

valid_files_names = os.listdir(valid_dir)
valid_files = [os.path.join(valid_dir, item) for item in valid_files_names
               if item.split("-")[0] == "train"]


# Parse TFRECORD and distort the image for train
def _parse_function(example_proto):
    features = {"image/encoded": tf.FixedLenFeature((), tf.string, default_value=''),
                "image/height": tf.FixedLenFeature((), tf.int64, default_value=[0]),
                "image/width": tf.FixedLenFeature((), tf.int64, default_value=[0]),
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
                                            lambda: (RESIZE_MIN, tf.cast(
                                                tf.multiply(tf.cast(width, tf.float64), tf.divide(RESIZE_MIN, height)),
                                                tf.int32)),
                                            lambda: (tf.cast(
                                                tf.multiply(tf.cast(height, tf.float64), tf.divide(RESIZE_MIN, width)),
                                                tf.int32), RESIZE_MIN))
    image_float = image_decoded#tf.image.convert_image_dtype(image_decoded, tf.float32)
    resized = tf.image.resize_images(image_float, [resized_height, resized_width])

    # Random crop from the resized image
    cropped = tf.random_crop(resized, [IMAGEHEIGHT, IMAGEWIDTH, 3])

    # # Flip to add a little more random distortion in.
    flipped = tf.image.random_flip_left_right(cropped)

    # # Standardization the image
    image_train = flipped
    # image_train = tf.image.per_image_standardization(flipped)
    features = {'images': image_train}

    # return image_train, tf.one_hot(parsed_features["label"][0], 1000)
    return features, parsed_features["image/class/label"][0]


def train_input_fn():
    dataset_train = tf.data.TFRecordDataset(train_files)
    dataset_train = dataset_train.map(_parse_function, num_parallel_calls=4)
    dataset_train = dataset_train.repeat(NB_EPOCH)
    dataset_train = dataset_train.batch(BATCH_SIZE)
    dataset_train = dataset_train.prefetch(BATCH_SIZE)
    return dataset_train

# # Initialize all tfrecord paths
# dataset = tf.data.TFRecordDataset(train_files)
# dataset = dataset.map(_parse_function)
# iterator = dataset.make_one_shot_iterator()
# next_image_data = iterator.get_next()
#
# imgs = next_image_data[0]["images"]
# lbs =  next_image_data[1]
# print(imgs)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     try:
#         i = 0
#         while True:
#             img, label = sess.run([imgs, lbs])
#
#             cv2.imwrite("./ImageNet/raw-data/{}.jpg".format(i), img)
#             i += 1
#             # print(data_record)
#             print(label)
#             # cv2.imshow("dkfdl", img)
#             # cv2.waitKey()
#     except:
#         pass

def _parse_test_function(example_proto):
    features = {"image/encoded": tf.FixedLenFeature((), tf.string, default_value=''),
                "image/height": tf.FixedLenFeature((), tf.int64, default_value=[0]),
                "image/width": tf.FixedLenFeature((), tf.int64, default_value=[0]),
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
                                            lambda: (RESIZE_MIN, tf.cast(
                                                tf.multiply(tf.cast(width, tf.float64), tf.divide(RESIZE_MIN, height)),
                                                tf.int32)),
                                            lambda: (tf.cast(
                                                tf.multiply(tf.cast(height, tf.float64), tf.divide(RESIZE_MIN, width)),
                                                tf.int32), RESIZE_MIN))
    # image_float = tf.image.convert_image_dtype(image_decoded, tf.float32)
    # image_resized = tf.image.resize_images(image_float, [resized_height, resized_width])
    #
    # # calculate how many to be center crop
    # shape = tf.shape(image_resized)
    # height, width = shape[0], shape[1]
    # amount_to_be_cropped_h = (height - imageHeight)
    # crop_top = amount_to_be_cropped_h // 2
    # amount_to_be_cropped_w = (width - imageWidth)
    # crop_left = amount_to_be_cropped_w // 2
    # image_cropped = tf.slice(image_resized, [crop_top, crop_left, 0], [imageHeight, imageWidth, -1])
    # image_valid = tf.image.per_image_standardization(image_cropped)
    image_float = image_decoded
    resized = tf.image.resize_images(image_float, [resized_height, resized_width])
    cropped = tf.random_crop(resized, [IMAGEHEIGHT, IMAGEWIDTH, 3])
    features = {'images': cropped}
    # return image_valid, tf.one_hot(parsed_features["label"][0], 1000)
    return features, parsed_features["image/class/label"][0]


def val_input_fn():
    dataset_valid = tf.data.TFRecordDataset(valid_files)
    dataset_valid = dataset_valid.map(_parse_test_function, num_parallel_calls=4)
    dataset_valid = dataset_valid.batch(BATCH_SIZE)
    dataset_valid = dataset_valid.prefetch(BATCH_SIZE)
    return dataset_valid


def predict_input_fn():
    dataset_valid = tf.data.TFRecordDataset(valid_files)
    dataset_valid = dataset_valid.map(_parse_test_function, num_parallel_calls=4)
    dataset_valid = dataset_valid.take(BATCH_SIZE)
    dataset_valid = dataset_valid.batch(BATCH_SIZE)
    return dataset_valid



def model_fn(features, labels, mode, params):

    # model = mobilenet_model_v2(IMAGEHEIGHT, IMAGEWIDTH)

    training = (mode == TFE.estimator.ModeKeys.TRAIN)
    images = tf.reshape(features["images"], [-1, IMAGEHEIGHT, IMAGEWIDTH, 3])
    logits = cifar10_inference(images, training, NB_CLASSES)#model(images, training)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == TFE.estimator.ModeKeys.PREDICT:
        return TFE.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                                           export_outputs={'classify': TFE.estimator.export.PredictOutput(predictions)})

    # Calculate Loss (for both TRAIN and EVAL modes)
    # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    loss = tf.losses.softmax_cross_entropy(tf.one_hot(labels, NB_CLASSES), logits)

    # m = tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true=labels, y_pred=logits)
    # accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])

    # Configure the Training Op (for TRAIN mode)
    if mode == TFE.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        boundaries = [5000, 60000, 80000]
        values = [0.1, 0.01, 0.001, 0.0001]
        learning_rate = tf.compat.v1.train.piecewise_constant(global_step, boundaries, values)
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss=loss, global_step=global_step)

        return TFE.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    m = tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true=labels, y_pred=logits)
    tf.summary.scalar('top-5_accuracy', m)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    tf.summary.scalar('accuracy', accuracy[0])

    eval_metric_ops = {"accuracy": accuracy}
    return TFE.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(_):
    if RUN_FROM_LAST == False:
        print("Removing previous artifacts...")
        shutil.rmtree(MODEL_NAME, ignore_errors=True)

    my_feature_columns = []
    my_feature_columns.append(tf.feature_column.numeric_column(key='images', shape=(IMAGEHEIGHT, IMAGEWIDTH, 3)))

    strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)

    run_config = TFE.estimator.RunConfig(save_checkpoints_steps=SAVE_CHECKPOINT_STEP,
                                         tf_random_seed=TF_RANDOM_SEED,
                                         model_dir=MODEL_NAME,
                                         save_summary_steps=1,
                                         train_distribute=strategy)
    imagenet_classifier = TFE.estimator.Estimator(model_fn=model_fn,
                                                  config=run_config)


    for _ in range(NB_EPOCH):
        imagenet_classifier.train(input_fn=train_input_fn, steps=NB_TRAIN_IMG // BATCH_SIZE)
        eval_results = imagenet_classifier.evaluate(input_fn=val_input_fn, steps=NB_VALID_IMG // BATCH_SIZE)
        print(eval_results)


if __name__ == "__main__":
    tf.app.run(main)
