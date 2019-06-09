from __future__ import print_function
import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist
import numpy as np

work_dir = os.getcwd()
mnist_dir = os.path.join(work_dir, "mnist")

if os.path.isdir(mnist_dir) == False:
    os.mkdir(mnist_dir)

save_dir = mnist_dir
datasets = mnist.read_data_sets(save_dir,
                                dtype=tf.uint8,
                                reshape=False,
                                validation_size=1000)

data_splits = ["train", "test", "validation"]

for d in range(len(data_splits)):
    print("saving: {}".format(data_splits[d]))
    data_set = datasets[d]

    file_name = os.path.join(save_dir, data_splits[d] + ".tfrecords")
    writer = tf.python_io.TFRecordWriter(file_name)

    for index in range(data_set.images.shape[0]):
        image = data_set.images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': tf.train.Feature(int64_list=
                                       tf.train.Int64List(value=
                                                          [data_set.images.shape[1]])),
            'width': tf.train.Feature(int64_list=
                                      tf.train.Int64List(value=
                                                         [data_set.images.shape[2]])),
            'depth': tf.train.Feature(int64_list=
                                      tf.train.Int64List(value=
                                                         [data_set.images.shape[3]])),
            'label': tf.train.Feature(int64_list=
                                      tf.train.Int64List(value=
                                                         [int(data_set.labels[index])])),
            'image_raw': tf.train.Feature(bytes_list=
                                          tf.train.BytesList(value=
                                                             [image]))}))

        writer.write(example.SerializeToString())
    writer.close()

# read TFRecord
filename = os.path.join(mnist_dir, 'train.tfrecords')
filename_queue = tf.train.string_input_producer([file_name], num_epochs=10)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example,
                                   features={'image_raw': tf.FixedLenFeature([], tf.string),
                                             'label': tf.FixedLenFeature([], tf.int64)})

image = tf.decode_raw(features['image_raw'], tf.uint8)
image.set_shape([784])
image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
label = tf.cast(features['label'], tf.int64)

# Randomly collect instances into batches
images_batch, labels_batch = tf.train.shuffle_batch(
    [image, label], batch_size=128,
    capacity=2000,
    min_after_dequeue=1000)

W = tf.get_variable("W", [28 * 28, 10])
y_pred = tf.matmul(images_batch, W)

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred,
                                                      labels=labels_batch)
NUM_EPOCHS = 10000
loss_mean = tf.reduce_mean(loss)
train_op = tf.train.AdamOptimizer().minimize(loss)

session = tf.Session()
init = tf.global_variables_initializer()
session.run([init])
init = tf.local_variables_initializer()
session.run(init)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=session, coord=coord)
print(threads)

try:
    step = 0
    while not coord.should_stop():
        step += 1
        session.run([train_op])
        if step % 100 == 0:
            loss_mean_val = session.run([loss_mean])
            print("step: {}, loss mean: {}".format(step, loss_mean_val))
except:
    print('Done training for %d epochs, %d steps.' % (NUM_EPOCHS, step))
finally:
    coord.request_stop()

coord.join(threads)
session.close()
