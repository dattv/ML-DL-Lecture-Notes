import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

root_path = os.path.dirname(os.path.dirname(__file__))

siamese_path = os.path.join(root_path, "siamese_tf")

mnist_path = os.path.join(siamese_path, "MNIST_data")
# if os.path.isdir(mnist_path) == False:
mnist_data = input_data.read_data_sets("MNIST_data", one_hot=True)

input_size = 784
no_classes = 10
batch_size = 1
total_batch = 3000

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

init_bias_value = 0.1
nChanel = 1
stddev = 0.1
def get_sub_model(input_):
    input_reshape = tf.reshape(input_, shape=[-1, 28, 28, 1], name="input_reshape")

    # convolution layer 1
    with tf.name_scope("conv1") as scope:
        with tf.name_scope("weights") as scope:
            weights_1 = tf.Variable(tf.truncated_normal([3, 3, nChanel, 64], stddev=stddev),
                                    name="weights_1")
        with tf.name_scope("biases") as scope:
            biases_1 = tf.Variable(tf.constant(init_bias_value, shape=[64]), name="biases")

        conv1 = tf.nn.conv2d(input=input_reshape, filter=weights_1, strides=[1, 1, 1, 1], padding="SAME")
        conv1 += biases_1

        conv1 = tf.nn.relu(conv1, name="conv1")

    # pooling layer 1
    with tf.name_scope("pooling1") as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")

    # convolution layer 2
    with tf.name_scope("conv2") as scope:
        with tf.name_scope("weights") as scope:
            weights_2 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=stddev), name="weights_2")
        with tf.name_scope("biases") as scope:
            biases_2 = tf.Variable(tf.constant(init_bias_value, shape=[128]), name="biases")

        conv2 = tf.nn.conv2d(input=pool1, filter=weights_2, strides=[1, 1, 1, 1], padding="SAME")
        conv2 += biases_2

        conv2 = tf.nn.relu(conv2, name="conv2")

    # pooling layer 2
    with tf.name_scope("pooling2") as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")

    # flatten layer 1
    size = pool2.shape
    size = int(size[1]) * int(size[2]) * int(size[3])

    flatten1 = tf.reshape(pool2, shape=[-1, size], name="flatten1")

    # fully layer 1
    with tf.name_scope("fully1") as scope:
        with tf.name_scope("weights") as scope:
            weights_full_1 = tf.Variable(tf.truncated_normal([size, 1024], stddev=stddev),
                                         name="weights_full_1")

        with tf.name_scope("biases") as scope:
            biases_full_1 = tf.Variable(tf.constant(init_bias_value, shape=[1024]), name="biases_full_1")

        fully1 = tf.matmul(flatten1, weights_full_1) + biases_full_1
        fully1 = tf.nn.relu(fully1, name="fully1")

    return fully1

def main():
    left_input = tf.placeholder(tf.float32, shape=[None, input_size])
    right_input = tf.placeholder(tf.float32, shape=[None, input_size])
    y_input = tf.placeholder(tf.float32, shape=[None, no_classes])
    dropout_bool = tf.placeholder(tf.bool)


    log_path = os.path.join(siamese_path, "temp")
    if os.path.isdir(log_path) == False:
        os.mkdir(log_path)

    with tf.variable_scope("siamese", reuse=True) as scope:
        left_bottle_neck = get_sub_model(left_input)
        right_bottle_neck =get_sub_model(right_input)

    dense_layer_bottle_neck = tf.concat([left_bottle_neck, right_bottle_neck], 1)

    dropout_layer = tf.layers.dropout(inputs=dense_layer_bottle_neck, rate=0.4, training=dropout_bool)

    with tf.name_scope("fully2") as scope:
        with tf.name_scope("weights") as scope:
            weights_full_2 = tf.Variable(tf.truncated_normal([2048, 10], stddev=stddev),
                                         name="weights_full_2")
        with tf.name_scope("biases") as scope:
            biases_full_2 = tf.Variable(tf.constant(init_bias_value, shape=[10]), name="biases_full_2")

        logits = tf.matmul(dropout_layer, weights_full_2) + biases_full_2
        logits = tf.nn.relu(logits, name="logits")

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
            accuracy_operation = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        merged_summary_operation = tf.summary.merge_all()
        train_summary_writer = tf.summary.FileWriter(os.path.join(log_path, "train"), session.graph)
        test_summary_writer = tf.summary.FileWriter(os.path.join(log_path, "test"))

        test_images, test_labels = mnist_data.test.images, mnist_data.test.labels

        for batch_no in range(total_batch):
            mnist_batch = mnist_data.train.next_batch(batch_size)

            train_images, train_labels = mnist_batch[0], mnist_batch[1]

            _, merged_summary = session.run([optimiser, merged_summary_operation],
                                            feed_dict={
                                                left_input: train_images,
                                                right_input: train_images,
                                                y_input: train_labels,
                                                dropout_bool: True
                                            })

            train_summary_writer.add_summary(merged_summary, batch_no)

            if batch_no % 1 == 0:
                merged_summary, _, err = session.run([merged_summary_operation,
                                                      accuracy_operation,
                                                      loss_operation], feed_dict={
                    left_input: test_images,
                    right_input: test_images,
                    y_input: test_labels,
                    dropout_bool: False
                })
                print("epoch: {}, error: {}".format(batch_no, err))
            test_summary_writer.add_summary(merged_summary, batch_no)


if __name__ == '__main__':
    main()
