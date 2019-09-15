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
        output_file = os.path.join(data_dir, mode+'.tfrecords')
        try:
          os.remove(output_file)
        except OSError:
          pass
        # Convert to tf.train.Example and write to TFRecords.
        _convert_to_tfrecord(input_files, output_file)

create_tfrecords_files()


class FLAGS():
  pass

FLAGS.batch_size = 200
FLAGS.max_steps = 1000
FLAGS.eval_steps = 100
FLAGS.save_checkpoints_steps = 100
FLAGS.tf_random_seed = 19851211
FLAGS.model_name = 'cnn-model-02'
FLAGS.use_checkpoint = False

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_DEPTH = 3
NUM_CLASSES = 10


def parse_record(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([IMAGE_DEPTH * IMAGE_HEIGHT * IMAGE_WIDTH])
    image = tf.reshape(image, [IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH])
    image = tf.cast(tf.transpose(image, [1, 2, 0]), tf.float32)

    label = tf.cast(features['label'], tf.int32)
    label = tf.one_hot(label, NUM_CLASSES)

    return image, label

def preprocess_image(image, is_training=False):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
        image, IMAGE_HEIGHT + 8, IMAGE_WIDTH + 8)

    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    image = tf.random_crop(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  return image


def generate_input_fn(file_names, mode=tf.estimator.ModeKeys.EVAL, batch_size=1):
    def _input_fn():
        dataset = tf.data.TFRecordDataset(filenames=file_names)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if is_training:
            buffer_size = batch_size * 2 + 1
            dataset = dataset.shuffle(buffer_size=buffer_size)

        # Transformation
        dataset = dataset.map(parse_record)
        dataset = dataset.map(
            lambda image, label: (preprocess_image(image, is_training), label))

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

  images = tf.feature_column.input_layer(
    features=features, feature_columns=feature_columns)

  images = tf.reshape(
    images, shape=(-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))

  # Calculate logits through CNN
  logits = inference(images)

  if mode in (tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL):
    predicted_indices = tf.argmax(input=logits, axis=1)
    probabilities = tf.nn.softmax(logits, name='softmax_tensor')

  if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
    global_step = tf.train.get_or_create_global_step()
    label_indices = tf.argmax(input=labels, axis=1)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=labels, logits=logits)
    tf.summary.scalar('cross_entropy', loss)

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'classes': predicted_indices,
        'probabilities': probabilities
    }
    export_outputs = {
        'predictions': tf.estimator.export.PredictOutput(predictions)
    }
    return tf.estimator.EstimatorSpec(
        mode, predictions=predictions, export_outputs=export_outputs)

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return tf.estimator.EstimatorSpec(
        mode, loss=loss, train_op=train_op)

  if mode == tf.estimator.ModeKeys.EVAL:
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(label_indices, predicted_indices)
    }
    return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=eval_metric_ops)

def serving_input_fn():
  receiver_tensor = {'images': tf.placeholder(
    shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], dtype=tf.float32)}
  features = {'images': tf.map_fn(preprocess_image, receiver_tensor['images'])}
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)


model_dir = 'trained_models/{}'.format(FLAGS.model_name)
train_data_files = ['cifar-10/train.tfrecords']
valid_data_files = ['cifar-10/validation.tfrecords']
test_data_files = ['cifar-10/eval.tfrecords']

run_config = tf.estimator.RunConfig(
  save_checkpoints_steps=FLAGS.save_checkpoints_steps,
  tf_random_seed=FLAGS.tf_random_seed,
  model_dir=model_dir
)

estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)

# There is another Exporter named FinalExporter
exporter = tf.estimator.LatestExporter(
  name='Servo',
  serving_input_receiver_fn=serving_input_fn,
  assets_extra=None,
  as_text=False,
  exports_to_keep=5)

train_spec = tf.estimator.TrainSpec(
  input_fn=generate_input_fn(file_names=train_data_files,
                             mode=tf.estimator.ModeKeys.TRAIN,
                             batch_size=FLAGS.batch_size),
  max_steps=FLAGS.max_steps)

eval_spec = tf.estimator.EvalSpec(
  input_fn=generate_input_fn(file_names=valid_data_files,
                             mode=tf.estimator.ModeKeys.EVAL,
                             batch_size=FLAGS.batch_size),
  steps=FLAGS.eval_steps, exporters=exporter)

if not FLAGS.use_checkpoint:
  print("Removing previous artifacts...")
  shutil.rmtree(model_dir, ignore_errors=True)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


test_input_fn = generate_input_fn(file_names=test_data_files,
                                  mode=tf.estimator.ModeKeys.EVAL,
                                  batch_size=1000)
estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)
print(estimator.evaluate(input_fn=test_input_fn, steps=1))

export_dir = model_dir + '/export/Servo/'
saved_model_dir = os.path.join(export_dir, os.listdir(export_dir)[-1])

predictor_fn = tf.contrib.predictor.from_saved_model(
  export_dir = saved_model_dir,
  signature_def_key='predictions')

import numpy

data_dict = _read_pickle_from_file('cifar-10/cifar-10-batches-py/test_batch')

N = 1000
images = data_dict['data'][:N].reshape([N, 3, 32, 32]).transpose([0, 2, 3, 1])
labels = data_dict['labels'][:N]

output = predictor_fn({'images': images})
accuracy = numpy.sum(
  [ans==ret for ans, ret in zip(labels, output['classes'])]) / float(N)

print(accuracy)


