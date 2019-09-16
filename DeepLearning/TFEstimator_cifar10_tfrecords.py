import logging
import os
import shutil
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_estimator as TFE
import pickle as cPickle
from DeepLearning.wide_resnet import WideResNet
from DeepLearning.cifar10_model import cifar10_inference

"""
method to get model_file name depending on file code
whole of model information including logging files, model-check-point files, saved-model files will be stored
in the model_file
this way will hep us to get rid of hard-coding
"""
file_name = __file__
file_name = os.path.split(file_name)[1]
file_name = file_name.split(".")[0]
model_file = "./" + file_name + "_model"
Path(model_file).mkdir(exist_ok=True)
tf.logging.set_verbosity(logging.INFO)
handlers = [logging.FileHandler(model_file + '/train.log'),
            logging.StreamHandler(sys.stdout)
            ]
logging.getLogger('tensorflow').handlers = handlers

print("tensorflow: {}".format(tf.__version__))
ROOT_PATH = os.getcwd()
print(ROOT_PATH)

"""
In this code the cifar10 data file already downloaded and stored in the folder: cifar-10-batches-py
if there is no cifar-10-batches-py folder in the same directory of this code, 
we have to download cifar10 dataset and attracts it to this the same directory of this code.
"""
DATA_DIR = "./cifar-10-batches-py"

CIFAR_FILENAME = 'cifar-10-python.tar.gz'
CIFAR_DOWNLOAD_URL = 'http://www.cs.toronto.edu/~kriz/' + CIFAR_FILENAME
CIFAR_LOCAL_FOLDER = 'cifar-10-batches-py'


def _download_and_extract(data_dir):
    tf.contrib.learn.datasets.base.maybe_download(CIFAR_FILENAME, data_dir, CIFAR_DOWNLOAD_URL)
    tarfile.open(os.path.join(data_dir, CIFAR_FILENAME), 'r:gz').extractall(data_dir)


def _get_file_names():
    """Returns the file names expected to exist in the input_dir."""
    file_names = {}
    file_names['train'] = ['data_batch_%d' % i for i in xrange(1, 5)]
    file_names['validation'] = ['data_batch_5']
    file_names['eval'] = ['test_batch']
    return file_names


def _read_pickle_from_file(filename):
    with tf.gfile.Open(file_name, "r") as f:
        data_dict = cPickle.load(f)

    return data_dict


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def _convert_to_tfrecord(input_files, output_file):
    print("generating: {}".format(output_file))

    with tf.python_io.TFRecordWriter(output_file) as record_writer:
        for input_file in input_files:
            data_dict = _read_pickle_from_file(input_file)
            data = data_dict["data"]
            labels = data_dict["labels"]
            num_entries_in_batch = len(data)
            for i in range(num_entries_in_batch):
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        "image": _bytes_feature(data[i].tobyte()),
                        "label": _int64_feature(labels[i])
                    }
                ))
                record_writer.write(example.SerializeToString())


def create_tfrecords_files(data_dir='cifar-10'):
    _download_and_extract(data_dir)
    file_names = _get_file_names()
    input_dir = os.path.join(data_dir, CIFAR_LOCAL_FOLDER)

    for mode, files in file_names.items():
        input_files = [os.path.join(input_dir, f) for f in files]
        output_file = os.path.join(data_dir, mode + '.tfrecords')
        try:
            os.remove(output_file)
        except OSError:
            pass
        # Convert to tf.train.Example and write to TFRecords.
        _convert_to_tfrecord(input_files, output_file)


# create_tfrecords_files()


class FLAGS():
    pass


FLAGS.batch_size = 4
FLAGS.max_steps = 1000
FLAGS.eval_steps = 100
FLAGS.save_checkpoints_steps = 100
FLAGS.tf_random_seed = 19851211
FLAGS.model_name = 'cnn-model-02'
FLAGS.use_checkpoint = False

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_DEPTH = 3
NUM_CLASSES = 1000


def parse_record(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/colorspace': tf.FixedLenFeature([], tf.string),
            'image/channels': tf.FixedLenFeature([], tf.int64),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/class/synset': tf.FixedLenFeature([], tf.string),
            'image/class/text': tf.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.FixedLenFeature([], tf.float32),
            'image/object/bbox/xmax': tf.FixedLenFeature([], tf.float32),
            'image/object/bbox/ymin': tf.FixedLenFeature([], tf.float32),
            'image/object/bbox/ymax': tf.FixedLenFeature([], tf.float32),
            'image/object/bbox/label': tf.FixedLenFeature([], tf.int64),
            'image/format': tf.FixedLenFeature([], tf.string),
            'image/filename': tf.FixedLenFeature([], tf.string),
            'image/encoded': tf.FixedLenFeature([], tf.string),
        })

    image = tf.decode_raw(features['image/encoded'], tf.int64)
    height = tf.cast(features["image/height"], tf.int64)
    width = tf.cast(features["image/width"], tf.int64)
    img_channesl = tf.cast(features["image/channels"], tf.int64)

    image_shape = tf.stack([height, width, 3])

    image = tf.reshape(image, image_shape)
    image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                   target_height=IMAGE_HEIGHT,
                                                   target_width=IMAGE_WIDTH)

    # image.set_shape([IMAGE_DEPTH * IMAGE_HEIGHT * IMAGE_WIDTH])
    # image = tf.reshape(image, [IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH])
    # image = tf.cast(tf.transpose(image, [1, 2, 0]), tf.float32)

    label = tf.cast(features['image/class/label'], tf.int32)
    label = tf.one_hot(label, NUM_CLASSES)

    return image, label


def preprocess_image(image, is_training=False):
    """Preprocess a single image of layout [height, width, depth]."""

    if is_training:
        # Resize the image to add four extra pixels on each side.
        image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_HEIGHT + 8, IMAGE_WIDTH + 8)

        # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
        image = tf.random_crop(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)
    return image


def generate_input_fn(file_names, mode=TFE.estimator.ModeKeys.EVAL, batch_size=1):
    def _input_fn():
        dataset = tf.data.TFRecordDataset(filenames=file_names)

        is_training = (mode == TFE.estimator.ModeKeys.TRAIN)
        if is_training:
            buffer_size = batch_size * 2 + 1
            dataset = dataset.shuffle(buffer_size=buffer_size)

        # Transformation
        dataset = dataset.map(parse_record)
        dataset = dataset.map(lambda image, label: (preprocess_image(image, is_training), label))

        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(2 * batch_size)

        images, labels = dataset.make_one_shot_iterator().get_next()

        features = {'images': images}
        return features, labels

    return _input_fn


def get_feature_columns():
    feature_columns = {
        'images': tf.feature_column.numeric_column('images', (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)),
    }
    return feature_columns


feature_columns = get_feature_columns()
print("Feature Columns: {}".format(feature_columns))


def model_fn(features, labels, mode, params):
    # Create the input layers from the features
    feature_columns = list(get_feature_columns().values())

    images = tf.feature_column.input_layer(features=features, feature_columns=feature_columns)

    images = tf.reshape(images, shape=(-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))

    # Calculate logits through CNN
    # logits = WideResNet(images, mode == TFE.estimator.ModeKeys.TRAIN, IMAGE_HEIGHT, nb_class=NUM_CLASSES, depth=16, k=8)()
    logits = cifar10_inference(images, mode == TFE.estimator.ModeKeys.TRAIN, NUM_CLASSES)

    if mode in (TFE.estimator.ModeKeys.PREDICT, TFE.estimator.ModeKeys.EVAL):
        predicted_indices = tf.argmax(input=logits, axis=1)
        probabilities = tf.nn.softmax(logits, name='softmax_tensor')

    if mode in (TFE.estimator.ModeKeys.TRAIN, TFE.estimator.ModeKeys.EVAL):
        global_step = tf.train.get_or_create_global_step()
        label_indices = tf.argmax(input=labels, axis=1)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        tf.summary.scalar('cross_entropy', loss)

    if mode == TFE.estimator.ModeKeys.PREDICT:
        predictions = {'classes': predicted_indices,
                       'probabilities': probabilities
                       }
        export_outputs = {'predictions': tf.estimator.export.PredictOutput(predictions)
                          }
        return TFE.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs)

    if mode == TFE.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return TFE.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == TFE.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(label_indices, predicted_indices)
        }
        return TFE.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops)


def serving_input_fn():
    receiver_tensor = {'images': tf.placeholder(shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], dtype=tf.float32)}
    features = {'images': tf.map_fn(preprocess_image, receiver_tensor['images'])}
    return TFE.estimator.export.ServingInputReceiver(features, receiver_tensor)


model_dir = 'trained_models/{}'.format(FLAGS.model_name)

TFRECORD_TRAIN_DIR = "/media/dat/68fa98f8-9d03-4c1e-9bdb-c71ea72ab6fa/dat/dataset/tfrecord/tfrecord_ImageNet"
TFRECORD_TEST_DIR = "/media/dat/68fa98f8-9d03-4c1e-9bdb-c71ea72ab6fa/dat/dataset/tfrecord/tfrecord_ImageNet"

train_data_files = [os.path.join(TFRECORD_TRAIN_DIR, f)
                    for f in os.listdir(TFRECORD_TRAIN_DIR) if f.split("-")[0] == "train"]

valid_data_files = [os.path.join(TFRECORD_TEST_DIR, f)
                    for f in os.listdir(TFRECORD_TEST_DIR) if f.split("-")[0] == "valid"]

valid_data_files = train_data_files
test_data_files = valid_data_files

run_config = TFE.estimator.RunConfig(save_checkpoints_steps=FLAGS.save_checkpoints_steps,
                                     tf_random_seed=FLAGS.tf_random_seed,
                                     model_dir=model_dir
                                     )

estimator = TFE.estimator.Estimator(model_fn=model_fn, config=run_config)

# There is another Exporter named FinalExporter
exporter = TFE.estimator.LatestExporter(name='Servo',
                                        serving_input_receiver_fn=serving_input_fn,
                                        assets_extra=None,
                                        as_text=False,
                                        exports_to_keep=5)

train_spec = TFE.estimator.TrainSpec(input_fn=generate_input_fn(file_names=train_data_files[0],
                                                                mode=TFE.estimator.ModeKeys.TRAIN,
                                                                batch_size=FLAGS.batch_size),
                                     max_steps=FLAGS.max_steps)

eval_spec = TFE.estimator.EvalSpec(input_fn=generate_input_fn(file_names=valid_data_files[0],
                                                              mode=TFE.estimator.ModeKeys.EVAL,
                                                              batch_size=FLAGS.batch_size),
                                   steps=FLAGS.eval_steps, exporters=exporter)

with tf.Session() as sess:
    feature = {
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/colorspace': tf.FixedLenFeature([], tf.string),
        'image/channels': tf.FixedLenFeature([], tf.int64),
        'image/class/label': tf.FixedLenFeature([], tf.int64),
        'image/class/synset': tf.FixedLenFeature([], tf.string),
        'image/class/text': tf.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.FixedLenFeature([], tf.float32),
        'image/object/bbox/xmax': tf.FixedLenFeature([], tf.float32),
        'image/object/bbox/ymin': tf.FixedLenFeature([], tf.float32),
        'image/object/bbox/ymax': tf.FixedLenFeature([], tf.float32),
        'image/object/bbox/label': tf.FixedLenFeature([], tf.int64),
        'image/format': tf.FixedLenFeature([], tf.string),
        'image/filename': tf.FixedLenFeature([], tf.string),
        'image/encoded': tf.FixedLenFeature([], tf.string),
    }
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([train_data_files[0]], num_epochs=1)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['image/encoded'], tf.float32)
    image = tf.decode_raw(features['image/encoded'], tf.int64)
    height = tf.cast(features["image/height"], tf.int64)
    width = tf.cast(features["image/width"], tf.int64)
    img_channesl = tf.cast(features["image/channels"], tf.int64)

    image_shape = tf.stack([height, width, 3])

    image = tf.reshape(image, image_shape)
    image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                   target_height=IMAGE_HEIGHT,
                                                   target_width=IMAGE_WIDTH)

    # Cast label data into int32
    label = tf.cast(features['image/class/label'], tf.int32)
    # Reshape image data into the original shape
    # image = tf.reshape(image, [224, 224, 3])

    # Any preprocessing here ...

    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1,
                                            min_after_dequeue=10)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for batch_index in range(5):
        img, lbl = sess.run([images, labels])
        img = img.astype(np.uint8)
        for j in range(6):
            plt.subplot(2, 3, j + 1)
            plt.imshow(img[j, ...])
            plt.title('cat' if lbl[j] == 0 else 'dog')
        plt.show()
    # Stop the threads
    coord.request_stop()

    # Wait for threads to stop
    coord.join(threads)
    sess.close()

if not FLAGS.use_checkpoint:
    print("Removing previous artifacts...")
    shutil.rmtree(model_dir, ignore_errors=True)

TFE.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
# for train_data_file in train_data_files:
#
#     train_spec = TFE.estimator.TrainSpec(input_fn=generate_input_fn(file_names=train_data_file,
#                                                                     mode=TFE.estimator.ModeKeys.TRAIN,
#                                                                     batch_size=FLAGS.batch_size),
#                                          max_steps=FLAGS.max_steps)
#
#     estimator.train(train_spec, steps=1)
