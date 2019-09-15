import logging
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow_estimator as TFE
import tensorflow as tf

from DeepLearning.cifar10_model import cifar10_inference
from DeepLearning.wide_resnet import WideResNet

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

"""
The code bellow will read the dataset file (binary format)
"""


def extract_data(index=0, filepath='./cifar-10-batches-bin/data_batch_5.bin'):
    print(os.path.isfile(filepath))
    bytestream = open(filepath, mode='rb')

    """
    In this type of dataset (only cifar10 dataset, the raw label data is numerical number
    such as 0, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 9
    (this is not one-hot type yet), so bytes length of raw label should be 1
    """
    label_bytes_length = 1

    """
    In this type of dataset (only cifar10 dataset, the raw image data is 32x32x3 numbers
    """
    image_bytes_length = 32 * 32 * 3

    """
    So the total bytes length should be summed of label bytes and image bytes
    """
    record_bytes_length = label_bytes_length + image_bytes_length

    bytestream.seek(record_bytes_length * index, 0)
    label_bytes = bytestream.read(label_bytes_length)
    image_bytes = bytestream.read(image_bytes_length)

    label = np.frombuffer(label_bytes, dtype=np.uint8)
    image = np.frombuffer(image_bytes, dtype=np.uint8)

    """
    Because we don't known what is the shape of image,
    it may be (32*32*3) or (32, 32, 3) so we have to reshape the image
    so that it always have shape like (32, 32, 3)
    """
    image = np.reshape(image, (3, 32, 32))
    image = np.transpose(image, [1, 2, 0])
    image = image.astype(np.float32) / 255.

    """"""
    result = {"image": image,
              "label": label}

    bytestream.close()
    return result


#
# result = extract_data(np.random.randint(1000))
# plt.imshow(result["image"])
# plt.show()


class FLAGS():
    pass


nb_epoch = 50
FLAGS.batch_size = 32
FLAGS.max_steps = int(50000 * nb_epoch // FLAGS.batch_size)
FLAGS.eval_steps = int(10000 // FLAGS.batch_size)
FLAGS.save_checkpoints_steps = int(50000 // FLAGS.batch_size)
FLAGS.tf_random_seed = 19851211
FLAGS.model_name = model_file
FLAGS.use_checkpoint = False

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_DEPTH = 3
NUM_CLASSES = 10
print("BATCH SIZE: {}".format(FLAGS.batch_size))
print("MAX STEP: {}".format(FLAGS.max_steps))
print("EVAL STEP: {}".format(FLAGS.eval_steps))
print("SAVE  CHECKPOINTS STEPS: {}".format(FLAGS.save_checkpoints_steps))
"""
Define input pipe line
"""


def parse_record(raw_record):
    label_bytes = 1
    image_bytes = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_DEPTH
    record_bytes = image_bytes + label_bytes

    record_vector = tf.decode_raw(raw_record, tf.uint8)

    label = tf.cast(record_vector[0], tf.int32)
    label = tf.one_hot(label, NUM_CLASSES)

    depth_major = tf.reshape(record_vector[label_bytes:record_bytes], [IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH])

    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    return image, label


def preprocess_image(image, is_training=False):
    """

    :param image:
    :param is_training:
    :return:
    """
    if is_training:
        image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_HEIGHT + np.random.randint(0, 8),
                                                       IMAGE_WIDTH + np.random.randint(0, 8))

        image = tf.random_crop(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])

        image = tf.image.random_flip_left_right(image)

        random_angles = tf.random.uniform(shape=(), minval=-np.pi / 4, maxval=np.pi / 4)
        image = tf.contrib.image.transform(
            image,
            tf.contrib.image.angles_to_projective_transforms(random_angles, tf.cast(tf.shape(image)[1], tf.float32),
                                                             tf.cast(tf.shape(image)[2], tf.float32))
        )

    image = tf.image.per_image_standardization(image)

    return image


def generate_input_fn(file_names, mode=TFE.estimator.ModeKeys.EVAL, batch_size=1):
    def _input_fn():
        label_bytes = 1
        image_bytes = IMAGE_DEPTH * IMAGE_HEIGHT * IMAGE_WIDTH
        record_bytes = label_bytes + image_bytes

        dataset = tf.data.FixedLengthRecordDataset(file_names,
                                                   record_bytes=record_bytes)

        is_training = (mode == TFE.estimator.ModeKeys.TRAIN)

        if is_training:
            buff_size = batch_size * 2 + 1

            dataset = dataset.shuffle(buffer_size=buff_size)

        dataset = dataset.map(parse_record)
        dataset = dataset.map(lambda image, label: (preprocess_image(image, is_training), label))

        dataset = dataset.repeat()
        dataset = dataset.prefetch(2 * batch_size)

        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        images, labels = iterator.get_next()

        features = {"images": images}
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
    """

    :param features:
    :param labels:
    :param mode:
    :param params:
    :return:
    """

    """
    Create the input layer from feature column
    """
    feature_columns = list(get_feature_columns().values())
    images = tf.feature_column.input_layer(features=features, feature_columns=feature_columns)

    images = tf.reshape(images, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])

    # logits = cifar10_inference(images, mode=mode == TFE.estimator.ModeKeys.TRAIN, NUM_CLASSES=NUM_CLASSES)
    logits = WideResNet(images, mode == TFE.estimator.ModeKeys.TRAIN, 32, nb_class=10)()

    # -----

    # if mode in (TFE.estimator.ModeKeys.PREDICT, TFE.estimator.ModeKeys.EVAL):
    predicted_indices = tf.argmax(logits, 1)
    probabilities = tf.nn.softmax(logits, name="Softmax_tensor")

    if mode in (TFE.estimator.ModeKeys.TRAIN, TFE.estimator.ModeKeys.EVAL):
        global_step = tf.train.get_or_create_global_step()

        label_indices = tf.argmax(labels, axis=1)
        eval_metric_ops = {"accuracy": tf.metrics.accuracy(label_indices, predicted_indices)}
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        # tf.summary.scalar("cross_entorpy", loss)

    if mode == TFE.estimator.ModeKeys.PREDICT:
        prediction = {"classes": predicted_indices,
                      "probability": probabilities}

        export_output = {"prediction": TFE.estimator.export.PredictOutput(prediction)}

        return TFE.estimator.EstimatorSpec(mode, predictions=prediction, export_outputs=export_output)

    if mode == TFE.estimator.ModeKeys.TRAIN:
        lr = tf.train.exponential_decay(1.e-2, global_step, 1000, 0.96, staircase=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)

        # tf.summary.scalar("accuracy", eval_metric_ops["accuracy"][1])
        tf.summary.scalar("learning_rate", lr)
        tf.summary.scalar("global_step", global_step)

        return TFE.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == TFE.estimator.ModeKeys.EVAL:
        return TFE.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)


def serving_input_fn():
    receiver_tensor = {'images': tf.placeholder(
        shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], dtype=tf.float32)}
    features = {'images': tf.map_fn(preprocess_image, receiver_tensor['images'])}

    return TFE.estimator.export.ServingInputReceiver(features, receiver_tensor)


model_dir = FLAGS.model_name
train_data_files = ['./cifar-10-batches-bin/data_batch_{}.bin'.format(i) for i in range(1, 5)]
valid_data_files = ['./cifar-10-batches-bin/data_batch_5.bin']
test_data_files = ['./cifar-10-batches-bin/test_batch.bin']

run_config = TFE.estimator.RunConfig(save_checkpoints_steps=FLAGS.save_checkpoints_steps,
                                     tf_random_seed=FLAGS.tf_random_seed,
                                     model_dir=FLAGS.model_name,
                                     save_summary_steps=1)

estimator = TFE.estimator.Estimator(model_fn=model_fn, config=run_config)

exporter = TFE.estimator.LatestExporter(name='Servo',
                                        serving_input_receiver_fn=serving_input_fn,
                                        assets_extra=None,
                                        as_text=False,
                                        exports_to_keep=5)

train_spec = TFE.estimator.TrainSpec(input_fn=generate_input_fn(file_names=train_data_files,
                                                                mode=TFE.estimator.ModeKeys.TRAIN,
                                                                batch_size=FLAGS.batch_size),
                                     max_steps=FLAGS.max_steps)

eval_spec = TFE.estimator.EvalSpec(input_fn=generate_input_fn(file_names=valid_data_files,
                                                              mode=TFE.estimator.ModeKeys.EVAL,
                                                              batch_size=FLAGS.batch_size),
                                   steps=FLAGS.eval_steps, exporters=exporter,
                                   throttle_secs=1)

if not FLAGS.use_checkpoint:
    print("Removing previous artifacts...")
    shutil.rmtree(model_dir, ignore_errors=True)

TFE.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

test_input_fn = generate_input_fn(file_names=test_data_files,
                                  mode=TFE.estimator.ModeKeys.EVAL,
                                  batch_size=1000)
estimator = TFE.estimator.Estimator(model_fn=model_fn, config=run_config)
print(estimator.evaluate(input_fn=test_input_fn, steps=1))

export_dir = model_dir + '/export/Servo/'
saved_model_dir = os.path.join(export_dir, os.listdir(export_dir)[-1])

predictor_fn = tf.contrib.predictor.from_saved_model(
    export_dir=saved_model_dir,
    signature_def_key='predictions')

N = 1000
labels = []
images = []

for i in range(N):
    result = extract_data(i, filepath='./cifar-10-batches-bin/test_batch.bin')
    images.append(result['image'])
    labels.append(result['label'][0])

output = predictor_fn({'images': images})

np.sum([a == r for a, r in zip(labels, output['classes'])]) / float(N)
