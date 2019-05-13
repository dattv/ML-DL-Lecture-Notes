import tensorflow as tf
import os

from tensorflow.examples.tutorials.mnist import input_data

LOG_DIR = "./tmp"
if os.path.isdir(LOG_DIR) == False:
    os.mkdir(LOG_DIR)

mnist_data = input_data.read_data_sets("MNIST_data", one_hot=True)

train_mnist_data = mnist_data.train
test_mnist_data = mnist_data.test
vali_mnist_data = mnist_data.validation




def add_variable_summary(tf_variable, summary_name):
    with tf.name_scope(summary_name + "_summary"):
        mean = tf.reduce_mean(tf_variable)
        tf.summary.scalar("Mean", mean)
        with tf.name_scope("standard_deviation"):
            standard_deviation = tf.sqrt(tf.reduce_mean(tf.square(tf_variable - mean)))

        tf.summary.scalar("StandardDeviation", standard_deviation)
        tf.summary.scalar("Maximum", tf.reduce_max(tf_variable))
        tf.summary.scalar("Minim", tf.reduce_min(tf_variable))
        tf.summary.scalar("Histogram", tf_variable)


def convolution_layer(input_layer, filters, kernel_size=[3, 3],
                      activation=tf.nn.relu):
    layer = tf.layers.conv2d(
        inputs=input_layer,
        filters=filters,
        kernel_size=kernel_size,
        activation=activation,
    )
    # add_variable_summary(layer, 'convolution')
    return layer


def pooling_layer(input_layer, pool_size=[2, 2], strides=2):
    layer = tf.layers.max_pooling2d(
        inputs=input_layer,
        pool_size=pool_size,
        strides=strides
    )
    # add_variable_summary(layer, 'pooling')
    return layer


def dense_layer(input_layer, units, activation=tf.nn.relu):
    layer = tf.layers.dense(
        inputs=input_layer,
        units=units,
        activation=activation)
    # add_variable_summary(layer, 'dense')
    return layer


HEIGHT = 28
WIDTH = 28
INPUT_SIZE = HEIGHT * WIDTH
NUM_CLASSES = train_mnist_data.labels.shape[1]

TOTAL_BATCH = 1000
BATCH_SIZE = 500
NCHANEL = 1

STDDEV_ = 0.1

x_input = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE], name='x_input')
y_input = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='y_input')

x_input_reshape = tf.reshape(x_input, shape=[-1, HEIGHT, WIDTH, NCHANEL], name='x_input_2d')

w1 = tf.Variable(tf.truncated_normal([3, 3, NCHANEL, 64], stddev=STDDEV_), name='w1')
b1 = tf.Variable(tf.constant(0., shape=[64]), name='b1')
conv_1 = tf.nn.conv2d(input=x_input_reshape,
                      filter=w1,
                      padding='VALID',
                      strides=[1, 1, 1, 1],
                      name='conv_1')
conv_1 += b1
convolution_layer_1 = tf.nn.relu(conv_1)
# convolution_layer_1 = convolution_layer(x_input_reshape, 64)

# pooling_layer_1 = pooling_layer(convolution_layer_1)
pooling_layer_1 = tf.nn.max_pool(convolution_layer_1,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='VALID',
                                 name="pooling_layer_1")

# convolution_layer_2 = convolution_layer(pooling_layer_1, 128)
w2 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=STDDEV_), name='w2')
b2 = tf.Variable(tf.constant(0., shape=[128]), name='b2')
convolution_layer_2 = tf.nn.conv2d(input=pooling_layer_1,
                                   filter=w2,
                                   padding='VALID',
                                   strides=[1, 1, 1, 1],
                                   name='convolution_layer_2')

# pooling_layer_2 = pooling_layer(convolution_layer_2)
pooling_layer_2 = tf.nn.max_pool(convolution_layer_2,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='VALID',
                                 name="pooling_layer_2")

# flattened_pool = tf.reshape(pooling_layer_2,
#                             [-1, 5 * 5 * 128],
#                             name='flattened_pool')
new_shape = pooling_layer_2.shape[1] * pooling_layer_2.shape[2] * pooling_layer_2.shape[3]
flattened_pool = tf.reshape(pooling_layer_2, [-1, new_shape], name="flattened_pool")

# dense_layer_bottleneck = dense_layer(flattened_pool, 1024)
w3 = tf.Variable(tf.truncated_normal(shape=[int(new_shape), 1024], stddev=STDDEV_), name='w3')
b3 = tf.Variable(tf.constant(0., shape=[1024]), name='b3')

dense_layer_bottleneck = tf.add(tf.matmul(flattened_pool, w3), b3)
dense_layer_bottleneck = tf.nn.relu(dense_layer_bottleneck)

dropout_bool = tf.placeholder(tf.bool)
dropout_layer = tf.layers.dropout(
    inputs=dense_layer_bottleneck,
    rate=0.4,
    training=dropout_bool
)

# logits = dense_layer(dropout_layer, NUM_CLASSES)
w4 = tf.Variable(tf.truncated_normal(shape=[1024, NUM_CLASSES], stddev=STDDEV_), name='w4')
b4 = tf.Variable(tf.constant(0., shape=[NUM_CLASSES]), name='b4')
logits = tf.add(tf.matmul(dropout_layer, w4), b4)
logits = tf.nn.relu(logits)

with tf.name_scope('loss'):
    softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=y_input, logits=logits)
    loss_operation = tf.reduce_mean(softmax_cross_entropy, name='loss')
    tf.summary.scalar('loss', loss_operation)

with tf.name_scope('optimiser'):
    optimizer = tf.train.AdamOptimizer().minimize(loss_operation)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        predictions = tf.argmax(logits, 1)
        correct_predictions = tf.equal(predictions, tf.argmax(y_input, 1))
    with tf.name_scope('accuracy'):
        accuracy_operation = tf.reduce_mean(
            tf.cast(correct_predictions, tf.float32))
tf.summary.scalar('accuracy', accuracy_operation)

session = tf.Session()
session.run(tf.global_variables_initializer())

file_name = os.path.basename(__file__)
file_name = file_name.split(".")
file_name = file_name[0]

merged_summary_operation = tf.summary.merge_all()
train_summary_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, file_name) + "/train", session.graph)
test_summary_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, file_name) + "/test")

test_images, test_labels = mnist_data.test.images, mnist_data.test.labels

for batch_no in range(TOTAL_BATCH):
    mnist_batch = mnist_data.train.next_batch(batch_size=BATCH_SIZE)

    train_images = mnist_batch[0]
    train_labels = mnist_batch[1]

    _, merged_summary = session.run([optimizer, merged_summary_operation], feed_dict={x_input: train_images,
                                                                                      y_input: train_labels,
                                                                                      dropout_bool: True})

    train_summary_writer.add_summary(merged_summary, batch_no)

    if batch_no % 100 == 0:
        acc, loss = session.run([accuracy_operation, loss_operation], feed_dict={x_input: test_images,
                                                                                 y_input: test_labels,
                                                                                 dropout_bool: False})
        print("batch no : {}, ACC: {}, LOSS: {}".format(batch_no, acc, loss))
        # test_summary_writer.add_summary(acc, batch_no)
