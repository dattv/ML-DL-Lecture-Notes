import pandas as pd
import logging
import argparse
from pathlib import Path
import numpy as np
# from keras.callbacks import LearningRateScheduler, ModelCheckpoint
# from keras.optimizers import SGD, Adam
from keras.utils import np_utils
# from wide_resnet import WideResNet
from wide_resnet_ import WideResNet
from utils import load_data
# from keras.preprocessing.image import ImageDataGenerator
from mixup_generator import MixupGenerator
from random_eraser import get_random_eraser

from tensorflow.python import debug as tf_debug

logging.basicConfig(level=logging.DEBUG)

import tensorflow as tf

LearningRateScheduler = tf.keras.callbacks.LearningRateScheduler
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
SGD = tf.keras.optimizers.SGD
Adam = tf.keras.optimizers.Adam
to_categorical = tf.keras.utils.to_categorical

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator


def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age and gender estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to input database mat file")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=30,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="initial learning rate")
    parser.add_argument("--opt", type=str, default="sgd",
                        help="optimizer name; 'sgd' or 'adam'")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network (should be 10, 16, 22, 28, ...)")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    parser.add_argument("--validation_split", type=float, default=0.1,
                        help="validation split ratio")
    parser.add_argument("--aug", action="store_true",
                        help="use data augmentation if set true")
    parser.add_argument("--output_path", type=str, default="checkpoints",
                        help="checkpoint dir")
    args = parser.parse_args()
    return args


class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.2
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.04
        return self.initial_lr * 0.008


def get_optimizer(opt_name, lr):
    if opt_name == "sgd":
        return SGD(lr=lr, momentum=0.9, nesterov=True)
    elif opt_name == "adam":
        return Adam(lr=lr)
    else:
        raise ValueError("optimizer name should be 'sgd' or 'adam'")


def main():
    # args = get_args()
    input_path = "./data/imdb_db.mat"  # args.input
    batch_size = 32  # args.batch_size
    nb_epochs = 30  # args.nb_epochs
    lr = 0.1  # args.lr
    opt_name = "sgd"  # args.opt
    depth = 16  # args.depth
    k = 8  # args.width
    validation_split = 0.1  # args.validation_split
    use_augmentation = False  # args.aug
    output_path = Path(__file__).resolve().parent.joinpath("./output")
    output_path.mkdir(parents=True, exist_ok=True)

    logging.debug("Loading data...")
    image, gender, age, _, image_size, _ = load_data(input_path)
    X_data = image
    y_data_g = to_categorical(gender, 2)
    y_data_a = to_categorical(age, 101)
    image_size = 64

    input = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name="input")
    pred_g, pred_a = WideResNet(input, image_size, depth=depth, k=k, is_train=True, is_train_able=True)()

    target_gender = tf.placeholder(tf.float32, shape=[None, 2], name="target_gender")
    target_age = tf.placeholder(tf.float32, shape=[None, 101], name="target_age")

    temp_loss_g = tf.keras.losses.categorical_crossentropy(target_gender, pred_g)
    temp_loss_a = tf.keras.losses.categorical_crossentropy(target_age, pred_a)

    pred_gender_loss = tf.math.reduce_mean(temp_loss_g)
    pred_age_loss = tf.math.reduce_mean(temp_loss_a)
    tf.summary.scalar("loss_g", pred_gender_loss)
    tf.summary.scalar("loss_a", pred_age_loss)

    total_loss = tf.math.add(pred_gender_loss, pred_age_loss)

    diff = tf.math.abs(tf.argmax(pred_a, 1) - tf.argmax(target_age, 1))
    age_acc = 1. - tf.reduce_mean(tf.cast(diff, dtype=tf.float32), 0)/100.
    tf.summary.scalar("age_acc", age_acc)

    gender_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred_g, 1), tf.argmax(target_gender, 1)), tf.float32))
    tf.summary.scalar("gender_acc", gender_acc)

    optimiser = tf.train.GradientDescentOptimizer(0.1).minimize(total_loss)

    logging.debug("Model summary...")
    # model.count_params()
    # model.summary()

    logging.debug("Running training...")

    data_num = len(X_data)
    indexes = np.arange(data_num)
    np.random.shuffle(indexes)
    X_data = X_data[indexes]
    y_data_g = y_data_g[indexes]
    y_data_a = y_data_a[indexes]
    train_num = int(data_num * (1 - validation_split))
    X_train = X_data[:train_num]
    X_test = X_data[train_num:]
    y_train_g = y_data_g[:train_num]
    y_test_g = y_data_g[train_num:]
    y_train_a = y_data_a[:train_num]
    y_test_a = y_data_a[train_num:]

    merged = tf.summary.merge_all()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        train_file_writer = tf.summary.FileWriter("./log/train", session.graph)
        test_file_writer = tf.summary.FileWriter("./log/test", session.graph)
        train_file_epoch_writer = tf.summary.FileWriter("./log/train_epoch", session.graph)
        test_file_epoch_writer = tf.summary.FileWriter("./log/test_epoch", session.graph)


        train_epoch = tf.Summary()
        test_epoch = tf.Summary()


        # sess = tf_debug.TensorBoardDebugWrapperSession(session, "dat-800G5H-800G5S:6064")
        # train_file_writer.add_summary(session)

        abc = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name="abc")
        efg = 2 * abc

        id = 0
        for epoch in range(nb_epochs):
            total_batch = int(len(X_train) / batch_size) - 1
            t_a_loss = 0
            t_g_loss = 0
            t_a_acc = 0
            t_g_acc = 0

            for batch_id in range(total_batch):
                first = batch_id * batch_size
                last = first + batch_size
                _, merged_, total_err, g_loss, age_loss, g_acc, a_acc = session.run(
                    [optimiser, merged, total_loss, pred_gender_loss, pred_age_loss, gender_acc, age_acc],
                    feed_dict={input: X_train[first:last],
                               target_age: y_train_a[first:last],
                               target_gender: y_train_g[first:last]})
                print(
                    "TRAIN>BATCH_ID: {:6d}, FIRST: {:6d}, LAST: {:6d}, TOTAL_ERROR: {:.10f}, AGE_LOSS: {:.10f}, GENDER_LOSS: {:.10f}, "
                    "AGE_ACC: {:.10f}, GENGER_ACC: {:.10f}".format(
                        batch_id, first, last, total_err, age_loss, g_loss, a_acc, g_acc
                    ))
                train_file_writer.add_summary(merged_, id)

                id += 1

                t_a_loss += age_loss
                t_g_loss += g_loss
                t_a_acc += a_acc
                t_g_acc += g_acc

            t_a_loss /= total_batch
            t_g_loss /= total_batch
            t_a_acc /= total_batch
            t_g_acc /= total_batch

            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print("SUMMATION: TRAINING")
            print("EPOCH: {:6d}, TOTAL_AGE_LOSS: {:.10f}, TOTAL_GENDER_LOSS: {:.10f}, TOTAL_AGE_ACC: {:.10f}, TOTAL_GENDER_ACC: {:.10f}"
                  .format(epoch, t_a_loss, t_g_loss, t_a_acc, t_g_acc))
            train_epoch.value.add(tag="total_loss_age", simple_value=t_a_loss)
            train_epoch.value.add(tag="total_loss_gender", simple_value=t_g_loss)
            train_epoch.value.add(tag="total_acc_age", simple_value=t_a_acc)
            train_epoch.value.add(tag="total_acc_gender", simple_value=t_g_acc)
            train_file_epoch_writer.add_summary(train_epoch, epoch)

            if epoch % 2 == 0:
                saver.save(session, "./ckpt_model/")
                total_batch = int(len(X_test) / batch_size) - 1

                t_a_loss = 0
                t_g_loss = 0
                t_a_acc = 0
                t_g_acc = 0
                for batch_id in range(total_batch):
                    first = batch_id * batch_size
                    last = first + batch_size
                    merged_, total_err, g_loss, age_loss, g_acc, a_acc = session.run(
                        [merged, total_loss, pred_gender_loss, pred_age_loss, gender_acc, age_acc],
                        feed_dict={input: X_test[first:last],
                                   target_age: y_test_a[first:last],
                                   target_gender: y_test_g[first:last]})
                    print(
                        "TEST>BATCH_ID: {:6d}, FIRST: {:6d}, LAST: {:6d}, TOTAL_ERROR: {:.10f}, AGE_LOSS: {:.10f}, "
                        "GENDER_LOSS: {:.10f}, AGE_ACC: {:.10f}, GENGER_ACC: {:.10f}".format(
                            batch_id, first, last, total_err, age_loss, g_loss, a_acc, g_acc
                        ))

                    t_a_loss += age_loss
                    t_g_loss += g_loss
                    t_a_acc += a_acc
                    t_g_acc += g_acc

                t_a_loss /= total_batch
                t_g_loss /= total_batch
                t_a_acc /= total_batch
                t_g_acc /= total_batch

                print(
                    ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                print("SUMMATION: TESTING")
                print("EPOCH: {:6d}, TOTAL_AGE_LOSS: {:.10f}, TOTAL_GENDER_LOSS: {:.10f}, TOTAL_AGE_ACC: {:.10f}, TOTAL_GENDER_ACC: {:.10f}"
                      .format(epoch, t_a_loss, t_g_loss, t_a_acc, t_g_acc))

                test_epoch.value.add(tag="total_loss_age", simple_value=t_a_loss)
                test_epoch.value.add(tag="total_loss_gender", simple_value=t_g_loss)
                test_epoch.value.add(tag="total_acc_age", simple_value=t_a_acc)
                test_epoch.value.add(tag="total_acc_gender", simple_value=t_g_acc)
                test_file_epoch_writer.add_summary(test_epoch, epoch)
        saver.save(session, "./ckpt_model/")

if __name__ == '__main__':
    main()
