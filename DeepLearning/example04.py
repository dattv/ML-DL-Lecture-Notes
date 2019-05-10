import tensorflow as tf
import os

# Load mnist dataset
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets("MNIST_data", one_hot=True)

train_mnist_data = mnist_data.train
test_mnist_data = mnist_data.test
valid_mnist_data = mnist_data.validation

LOG_DIR = "./tmp"
if os.path.isdir(LOG_DIR) == False:
    os.mkdir(LOG_DIR)

def add_variable_summary(tf_variable, summary_name):
  with tf.name_scope(summary_name + '_summary'):
    mean = tf.reduce_mean(tf_variable)
    tf.summary.scalar('Mean', mean)
    with tf.name_scope('standard_deviation'):
        standard_deviation = tf.sqrt(tf.reduce_mean(
            tf.square(tf_variable - mean)))
    tf.summary.scalar('StandardDeviation', standard_deviation)
    tf.summary.scalar('Maximum', tf.reduce_max(tf_variable))
    tf.summary.scalar('Minimum', tf.reduce_min(tf_variable))
    tf.summary.histogram('Histogram', tf_variable)


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



height = 28
width = 28
input_size = height * width
no_classes = train_mnist_data.labels.shape[1]

total_batches = 200
batch_size = 500

x_input = tf.placeholder(tf.float32, shape=[None, input_size], name='x_input')
y_input = tf.placeholder(tf.float32, shape=[None, no_classes], name='y_input')

x_input_reshape = tf.reshape(x_input, [-1, 28, 28, 1], name='input_reshape')
convolution_layer_1 = convolution_layer(x_input_reshape, 64)
pooling_layer_1 = pooling_layer(convolution_layer_1)
convolution_layer_2 = convolution_layer(pooling_layer_1, 128)
pooling_layer_2 = pooling_layer(convolution_layer_2)
flattened_pool = tf.reshape(pooling_layer_2, [-1, 5 * 5 * 128],
                            name='flattened_pool')
dense_layer_bottleneck = dense_layer(flattened_pool, 1024)

dropout_bool = tf.placeholder(tf.bool)
dropout_layer = tf.layers.dropout(
        inputs=dense_layer_bottleneck,
        rate=0.4,
        training=dropout_bool
    )

logits = dense_layer(dropout_layer, no_classes)

with tf.name_scope('loss'):
    softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=y_input, logits=logits)
    loss_operation = tf.reduce_mean(softmax_cross_entropy, name='loss')
    tf.summary.scalar('loss', loss_operation)

with tf.name_scope('optimiser'):
    optimiser = tf.train.AdamOptimizer().minimize(loss_operation)


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
name = file_name[0]

merged_summary_operation = tf.summary.merge_all()
train_summary_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, name) + "/train", session.graph)
test_summary_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, name) + "/test")

test_images, test_labels = mnist_data.test.images, mnist_data.test.labels

for batch_no in range(total_batches):
    mnist_batch = mnist_data.train.next_batch(batch_size)
    train_images, train_labels = mnist_batch[0], mnist_batch[1]
    _, merged_summary = session.run([optimiser, merged_summary_operation],
                                    feed_dict={
        x_input: train_images,
        y_input: train_labels,
        dropout_bool: True
    })
    train_summary_writer.add_summary(merged_summary, batch_no)
    if batch_no % 10 == 0:
        merged_summary, acc, loss = session.run([merged_summary_operation,
                                         accuracy_operation, loss_operation], feed_dict={
            x_input: test_images,
            y_input: test_labels,
            dropout_bool: False
        })
        print("Epoch: {}, Accuracy: {}, Erro: {}".format(batch_no, acc, loss))
    test_summary_writer.add_summary(merged_summary, batch_no)