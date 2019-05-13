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

HEIGHT = 28
WIDTH = 28
INPUT_SIZE = HEIGHT * WIDTH
NUM_CLASSES = train_mnist_data.labels.shape[1]

TOTAL_BATCH = 200
BATCH_SIZE = 500
NCHANEL = 1

x_input = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE], name='x_input')
y_input = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='y_input')

x_input_reshape = tf.reshape(x_input, shape=[-1, HEIGHT, WIDTH, NCHANEL], name='x_input_2d')

w1 = tf.Variable(tf.truncated_normal([3, 3, NCHANEL, 64], stddev=0.1), name='w1')
b1 = tf.Variable(tf.constant(0.1, shape=[64]), name='b1')
conv_1 = tf.nn.conv2d(input=x_input_reshape,
                      filter=w1,
                      padding='VALID',
                      strides=[1, 1, 1, 1],
                      name='conv_1')
conv_1 += b1
conv_1 = tf.nn.relu(conv_1)

maxpool1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name="pool1")

w2 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1), name='w2')
b2 = tf.Variable(tf.constant(0.1, shape=[128]), name='b2')
conv_2 = tf.nn.conv2d(input=maxpool1,
                      filter=w2,
                      padding='VALID',
                      strides=[1, 1, 1, 1],
                      name='conv_2')

conv_2 += b2
conv_2 = tf.nn.relu(conv_2)

maxpool2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

new_shape = maxpool2.shape[1] * maxpool2.shape[2] * maxpool2.shape[3]

flattened = tf.reshape(maxpool2, shape=[-1, new_shape], name='flat')

w3 = tf.Variable(tf.truncated_normal(shape=[int(new_shape), 1024], stddev=0.1), name='w3')
b3 = tf.Variable(tf.constant(0.1, shape=[1024]), name='b3')

dense_layer_bottleneck = tf.add(tf.matmul(flattened, w3), b3)


keep_prob = tf.placeholder(tf.float32)
dropout_layer = tf.nn.dropout(dense_layer_bottleneck, rate=1. - keep_prob)

w4 = tf.Variable(tf.truncated_normal(shape=[1024, NUM_CLASSES], stddev=0.1), name='w4')
b4 = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name='b4')
logits = tf.add(tf.matmul(dropout_layer, w4), b4)

with tf.name_scope("loss") as scope:
    softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_input,
                                                                    logits=logits)
    loss_operation = tf.reduce_min(softmax_cross_entropy, name='loss')

with tf.name_scope("optimizer") as scope:
    optimizer = tf.train.AdamOptimizer().minimize(loss_operation)

with tf.name_scope("accuracy") as scope:
    with tf.name_scope("correct_predictions") as scope:
        pred = tf.argmax(logits, 1)
        correct_pred = tf.equal(pred, tf.argmax(y_input, 1))

with tf.Session() as session:
    file_name = os.path.basename(__file__)
    file_name = file_name.split(".")
    name = file_name[0]

    merged_summary_operation = tf.summary.merge_all()
    train_summary_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, name) + "/train", session.graph)
    test_summary_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, name) + "/test")

    test_images, test_labels = mnist_data.test.images, mnist_data.test.labels

    session.run(tf.global_variables_initializer())

    for batch_no in range(TOTAL_BATCH):
        mnist_batch = train_mnist_data.next_batch(BATCH_SIZE)
        train_img = mnist_batch[0]
        train_labels = mnist_batch[1]

        _, merged = session.run([optimizer, merged_summary_operation],
                                feed_dict={x_input: train_img,
                                           y_input: train_labels,
                                           keep_prob: 0.4})
        train_summary_writer.add_summary(merged, batch_no)

        if batch_no % 100 == 0:
            test_img = test_mnist_data.images
            test_labels = test_mnist_data.labels

            loss = session.run([loss_operation],
                               feed_dict={x_input: test_img,
                                          y_input: test_labels,
                                          keep_prob: 1.})

            print("EPOCH: {}, LOSS: {}".format(batch_no, loss))





