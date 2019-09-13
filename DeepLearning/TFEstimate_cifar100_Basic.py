
import logging
import os
import shutil
from pathlib import Path
import numpy as np
import sys
import tensorflow_estimator as TFE
import tensorflow as tf
import tarfile
from tqdm import tqdm
from urllib.request import urlretrieve
import pickle

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

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_DEPTH = 3
NUM_CLASSES = 100

def my_hook(t):
    """
    Wraps tqdm instance
    :param t:
    :return:
    """
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """

        :param b:       int option
        :param bsize:   int
        :param tsize:
        :return:
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


class cifar100:
    def __init__(self, cifar100_dir=None):
        self.name = "Cifar100 dataset"
        self.n_channels = 3
        self.width = 32
        self.heigh = 32
        self.DIR = "./cifar100_dataset"
        if os.path.isdir(self.DIR) == False:
            os.mkdir(self.DIR)

        cifar100_url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

        name = cifar100_url.split("/")
        name = name[len(name) - 1]

        full_file_path = os.path.join(self.DIR, name)

        if os.path.isfile(full_file_path) == False:
            print("downloading from: {}".format(cifar100_url))
            with tqdm(unit="B", unit_scale=True, leave=True, miniters=1, desc=cifar100_url.split("/")[-1]) as t:
                urlretrieve(cifar100_url, filename=full_file_path, reporthook=my_hook(t), data=None)
            print("finish download")

        # extract compressed file
        tar = tarfile.open(full_file_path)
        sub_folders = tar.getnames()
        train_subfolder = [f for f in sub_folders if os.path.split(f)[1] == "train"]
        self._train_subfolder = train_subfolder[0]
        test_subfolder = [f for f in sub_folders if os.path.split(f)[1] == "test"]
        self._test_subfolder = test_subfolder[0]
        tar.extractall()
        tar.close()

    def get_data(self):

        with open(self._train_subfolder, 'rb') as fo:
            try:
                self.samples_train = pickle.load(fo)
            except UnicodeDecodeError:  # python 3.x
                fo.seek(0)
                self.samples_train = pickle.load(fo, encoding='latin1')

        with open(self._test_subfolder, 'rb') as fo:
            try:
                self.samples_test = pickle.load(fo)
            except UnicodeDecodeError:
                fo.seek(0)
                self.samples_test = pickle.load(fo, encoding='latin1')

        train_data_img = self.samples_train["data"]
        train_data_labels = self.samples_train["fine_labels"]
        test_data_img = self.samples_test["data"]
        test_data_lables = self.samples_test["fine_labels"]

        return (train_data_img, train_data_labels), (test_data_img, test_data_lables)

dataset = cifar100()
(train_data_img, train_data_labels), (test_data_img, test_data_labels) = dataset.get_data()
train_img_number = len(train_data_img)
test_img_number = len(test_data_img)
print(dataset.name)
print("Number of Images in train set: {}".format(train_img_number))
print("Number of Images in test set: {}".format(test_img_number))
print("{}".format(train_data_img.shape))
print("Images size: {}x{}x{}".format(dataset.heigh, dataset.width, dataset.n_channels))

train_data_labels_one_hot = np.zeros((train_img_number, NUM_CLASSES))
train_data_labels_one_hot[np.arange(train_img_number), train_data_labels] = 1

test_data_labels_one_hot = np.zeros((test_img_number, NUM_CLASSES))
test_data_labels_one_hot[np.arange(test_img_number), test_data_labels] = 1

ROOT_PATH = os.getcwd()
print(ROOT_PATH)






def inference(images, mode):
    input_layer = tf.reshape(images, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])

    output1 = tf.layers.conv2d(input_layer, 64, 3, strides=1, padding='same', activation=None, name="conv1-1")
    output2 = tf.layers.batch_normalization(output1, training=mode == TFE.estimator.ModeKeys.TRAIN)
    relu2 = tf.nn.relu(output2)
    dropout1 = tf.layers.dropout(relu2, rate=0.4, training=mode == TFE.estimator.ModeKeys.TRAIN)

    output3 = tf.layers.conv2d(dropout1, 64, 3, strides=(1, 1), padding='same', activation=None, name="conv1-2")
    output4 = tf.layers.batch_normalization(output3, training=mode==TFE.estimator.ModeKeys.TRAIN)
    relu4 = tf.nn.relu(output4)
    dropout4 = tf.layers.dropout(relu4, rate=0.4, training=mode == TFE.estimator.ModeKeys.TRAIN)
    output5 = tf.layers.max_pooling2d(dropout4, pool_size=2, strides=2, padding="valid", name="pool1")
    #     print(output)

    output6 = tf.layers.conv2d(output5, 128, 3, strides=(1, 1), padding='same', activation=None, name="conv2-1")
    output7 = tf.layers.batch_normalization(output6, training=mode == TFE.estimator.ModeKeys.TRAIN)
    relu7 = tf.nn.relu(output7)
    dropout7 = tf.layers.dropout(relu7, rate=0.4, training=mode == TFE.estimator.ModeKeys.TRAIN)

    output8 = tf.layers.conv2d(dropout7, 128, 3, strides=(1, 1), padding='same', activation=None, name="conv2-2")
    output9 = tf.layers.batch_normalization(output8, training=mode == TFE.estimator.ModeKeys.TRAIN)
    relu9 = tf.nn.relu(output9)
    dropout9 = tf.layers.dropout(relu9, rate=0.4, training=mode == TFE.estimator.ModeKeys.TRAIN)
    output10 = tf.layers.max_pooling2d(dropout9, pool_size=2, strides=2, padding="valid", name="pool2")
    #     print(output)

    output11 = tf.layers.conv2d(output10, 256, 3, strides=(1, 1), padding='same', activation=None, name="conv3-1")
    output12 = tf.layers.batch_normalization(output11, training=mode == TFE.estimator.ModeKeys.TRAIN)
    relu12 = tf.nn.relu(output12)
    dropout12 = tf.layers.dropout(relu12, rate=0.4, training=mode == TFE.estimator.ModeKeys.TRAIN)

    output13 = tf.layers.conv2d(dropout12, 256, 3, strides=(1, 1), padding='same', activation=None, name="conv3-2")
    output14 = tf.layers.batch_normalization(output13, training=mode == TFE.estimator.ModeKeys.TRAIN)
    relu14 = tf.nn.relu(output14)
    dropout14 = tf.layers.dropout(relu14, rate=0.4, training=mode == TFE.estimator.ModeKeys.TRAIN)
    output15 = tf.layers.conv2d(dropout14, 256, 3, strides=(1, 1), padding='same', activation=None, name="conv3-3")

    output16 = tf.layers.max_pooling2d(output15, pool_size=2, strides=2, padding="valid", name="pool3")
    relu16 = tf.nn.relu(output16)
    dropout16 = tf.layers.dropout(relu16, rate=0.4, training=mode == TFE.estimator.ModeKeys.TRAIN)
    #     print(output)

    output17 = tf.layers.conv2d(dropout16, 512, 3, strides=(1, 1), padding='same', activation=None, name="conv4-1")
    output18 = tf.layers.batch_normalization(output17, training=mode == TFE.estimator.ModeKeys.TRAIN)
    relu18 = tf.nn.relu(output18)
    dropout18 = tf.layers.dropout(relu18, rate=0.4, training=mode == TFE.estimator.ModeKeys.TRAIN)

    output19 = tf.layers.conv2d(dropout18, 512, 3, strides=(1, 1), padding='same', activation=None, name="conv4-2")
    output20 = tf.layers.batch_normalization(output19, training=mode == TFE.estimator.ModeKeys.TRAIN)
    relu20 = tf.nn.relu(output20)
    dropout20 = tf.layers.dropout(relu20, rate=0.4, training=mode == TFE.estimator.ModeKeys.TRAIN)

    output21 = tf.layers.conv2d(dropout20, 512, 3, strides=(1, 1), padding='same', activation=tf.nn.relu, name="conv4-3")
    output22 = tf.layers.max_pooling2d(output21, pool_size=2, strides=2, padding="valid", name="pool4")
    #     print(output)

    output23 = tf.layers.conv2d(output22, 512, 3, strides=(1, 1), padding='same', activation=None, name="conv5-1")
    output24 = tf.layers.batch_normalization(output23, training=mode == TFE.estimator.ModeKeys.TRAIN)
    relu24 = tf.nn.relu(output24)
    dropout24 = tf.layers.dropout(relu24, rate=0.4, training=mode == TFE.estimator.ModeKeys.TRAIN)

    output25 = tf.layers.conv2d(dropout24, 512, 3, strides=(1, 1), padding='same', activation=None, name="conv5-2")
    output26 = tf.layers.batch_normalization(output25, training=mode == TFE.estimator.ModeKeys.TRAIN)
    relu26 = tf.nn.relu(output26)
    dropout26 = tf.layers.dropout(relu26, rate=0.4, training=mode == TFE.estimator.ModeKeys.TRAIN)
    output27 = tf.layers.conv2d(dropout26, 512, 3, strides=(1, 1), padding='same', activation=tf.nn.relu, name="conv5-3")

    output28 = tf.layers.max_pooling2d(output27, pool_size=2, strides=2, padding="valid", name="pool5")
    #     print(output)

    output_shape = output28.shape
    output29 = tf.reshape(output28, shape=[-1, int(output_shape[1] * output_shape[2] * output_shape[3])], name="flatten")
    output30 = tf.layers.dense(output29, 4096, activation=None, name="fully1")
    relu30 = tf.nn.relu(output30)
    dropout30 = tf.layers.dropout(relu30, rate=0.4, training=mode == TFE.estimator.ModeKeys.TRAIN)

    output31 = tf.layers.batch_normalization(dropout30, training=mode == TFE.estimator.ModeKeys.TRAIN)
    output32 = tf.layers.dense(output31, 4096, activation=None, name="fully2")
    relu32 = tf.nn.relu(output32)
    dropout32 = tf.layers.dropout(relu32, rate=0.4, training=mode == TFE.estimator.ModeKeys.TRAIN)

    output33 = tf.layers.dense(dropout32, NUM_CLASSES, activation=None, name="output")

    return output33


def cnn_model_fn(features, labels, mode):
    images = features["x"]
    with tf.variable_scope("model") as scope:
        logits = inference(images, mode)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }


    if mode in (TFE.estimator.ModeKeys.TRAIN, TFE.estimator.ModeKeys.EVAL):

        global_step = tf.train.get_global_step()
        labels_indices = tf.argmax(labels, axis=1)
        eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels_indices,
                                                           predictions=predictions["classes"])}

        loss = tf.losses.softmax_cross_entropy(labels, logits)

    if mode == TFE.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1.e-3).minimize(loss, global_step=global_step)

        return TFE.estimator.EstimatorSpec(mode, loss=loss, train_op=optimizer, eval_metric_ops=eval_metric_ops)

    if mode == TFE.estimator.ModeKeys.EVAL:
        return TFE.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    if mode == TFE.estimator.ModeKeys.PREDICT:

        export_outputs = {'predictions': TFE.estimator.export.PredictOutput(predictions)}
        return TFE.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)


# Load training and eval data

train_data = train_data_img.astype(np.float32)
eval_data = test_data_img.astype(np.float32)
train_labels = train_data_labels_one_hot
eval_labels = test_data_labels_one_hot

class FLAGS():
    pass


FLAGS.batch_size = 200
FLAGS.max_steps = None
FLAGS.nb_epoch = 100
FLAGS.eval_steps = int(len(eval_data) // FLAGS.batch_size)
FLAGS.save_checkpoints_steps = int(len(train_data) // FLAGS.batch_size)
FLAGS.tf_random_seed = 19851211
FLAGS.model_name = model_file
FLAGS.use_checkpoint = False

run_config = TFE.estimator.RunConfig(save_checkpoints_steps=FLAGS.save_checkpoints_steps,
                                     tf_random_seed=FLAGS.tf_random_seed,
                                     model_dir=FLAGS.model_name
                                     )

estimator = TFE.estimator.Estimator(model_fn=cnn_model_fn, config=run_config)
# Set up logging for predictions
tensors_to_log = {"accuracy": "accuracy"}

logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)


def preprocess_image(image, is_training=False):
    """Preprocess a single image of layout [height, width, depth]."""
    if is_training:
        # Resize the image to add four extra pixels on each side.
        image = tf.image.resize_image_with_crop_or_pad(
            image, IMAGE_HEIGHT + 8, IMAGE_WIDTH + 8)

        # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
        image = tf.random_crop(image, [IMAGE_HEIGHT, IMAGE_WIDTH, 1])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)
    return image


def serving_input_fn():
    receiver_tensor = {'x': tf.placeholder(
        shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH], dtype=tf.float32)}
    features = {'x': tf.map_fn(preprocess_image, receiver_tensor['x'])}

    return TFE.estimator.export.ServingInputReceiver(features, receiver_tensor)


# There is another Exporter named FinalExporter
exporter = TFE.estimator.LatestExporter(name='Servo',
                                        serving_input_receiver_fn=serving_input_fn,
                                        assets_extra=None,
                                        as_text=False,
                                        exports_to_keep=5)

train_input_fn = TFE.estimator.inputs.numpy_input_fn(x={"x": train_data},
                                                     y=train_labels,
                                                     batch_size=FLAGS.batch_size,
                                                     num_epochs=None,
                                                     shuffle=True,
                                                     num_threads=8)
train_spec = TFE.estimator.TrainSpec(input_fn=train_input_fn,
                                     max_steps=int(FLAGS.nb_epoch * len(train_data) // FLAGS.batch_size))

eval_input_fn = TFE.estimator.inputs.numpy_input_fn(x={"x": eval_data},
                                                    y=eval_labels,
                                                    num_epochs=1,
                                                    shuffle=False,
                                                    num_threads=8)
eval_spec = TFE.estimator.EvalSpec(input_fn=eval_input_fn,
                                   steps=FLAGS.eval_steps,
                                   exporters=exporter,
                                   throttle_secs=1)

if not FLAGS.use_checkpoint:
    print("Removing model dir")
    shutil.rmtree(model_file, ignore_errors=True)

TFE.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

