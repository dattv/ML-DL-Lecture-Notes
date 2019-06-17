import os

import tensorflow as tf


class siamese():
    def __init__(self):
        self.stddev_ = 0.1

    def make_model(self, input_tensor1, input_tensor2):
        encoded_l = self.sub_model(input_tensor1)
        encoded_r = self.sub_model(input_tensor2)

        l1_distance = tf.abs(encoded_l - encoded_r, name="L1_DISTANCE")

        with tf.name_scope("fully_layer_2") as scope:
            with tf.name_scope("weights") as scope:
                w_flat2_1 = tf.Variable(tf.truncated_normal([4096, 1], stddev=self.stddev_), name="w_flat2_1")
            with tf.name_scope("biases") as scope:
                b_flat2_1 = tf.Variable(tf.constant(0.1, shape=[1]), name="b_flat2_1")

            pred = tf.nn.sigmoid(tf.matmul(l1_distance, w_flat2_1) + b_flat2_1, name="pred")

        return pred

    def sub_model(self, input_tensor):
        n_chanel = int(input_tensor.shape[3])
        stddev_ = self.stddev_

        with tf.name_scope("conv_layer_1") as scope:
            with tf.name_scope("weights") as scope:
                w1_1 = tf.Variable(tf.truncated_normal([10, 10, n_chanel, 64], stddev=stddev_), name="w1_1")
            with tf.name_scope("biases") as scope:
                b1_1 = tf.Variable(tf.constant(0.1, shape=[64]), name="b1_1")

            conv1_1 = tf.nn.conv2d(input=input_tensor,
                                   filter=w1_1,
                                   strides=[1, 1, 1, 1],
                                   padding="SAME")
            conv1_1 += b1_1
            conv1_1 = tf.nn.relu(conv1_1, name="CONV1_1")

        with tf.name_scope("pooling_layer_1") as scope:
            pool1_1 = tf.nn.max_pool(conv1_1,
                                     ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1],
                                     padding="SAME",
                                     name="POOL1_1")

        with tf.name_scope("conv_layer_2") as scope:
            with tf.name_scope("weights") as scope:
                w2_1 = tf.Variable(tf.truncated_normal([7, 7, 64, 128], stddev=stddev_), name="w2_1")
            with tf.name_scope("biases") as scope:
                b2_1 = tf.Variable(tf.constant(0.1, shape=[128]), name="b2_1")

            conv2_1 = tf.nn.conv2d(input=pool1_1,
                                   filter=w2_1,
                                   strides=[1, 1, 1, 1],
                                   padding="SAME")
            conv2_1 += b2_1
            conv2_1 = tf.nn.relu(conv2_1, name="CONV2_1")

        with tf.name_scope("pooling_layer_2") as scope:
            pool2_1 = tf.nn.max_pool(conv2_1,
                                     ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1],
                                     padding="SAME",
                                     name="POOL2_1")

        with tf.name_scope("conv_layer_3") as scope:
            with tf.name_scope("weights") as scope:
                w3_1 = tf.Variable(tf.truncated_normal([4, 4, 128, 128], stddev=stddev_), name="w3_1")
            with tf.name_scope("biases") as scope:
                b3_1 = tf.Variable(tf.constant(0.1, shape=[128]), name="b3_1")

            conv3_1 = tf.nn.conv2d(input=pool2_1,
                                   filter=w3_1,
                                   strides=[1, 1, 1, 1],
                                   padding="SAME")
            conv3_1 += b3_1
            conv3_1 = tf.nn.relu(conv3_1, name="CONV3_1")

        with tf.name_scope("pooling_layer_3") as scope:
            pool3_1 = tf.nn.max_pool(conv3_1,
                                     ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1],
                                     padding="SAME",
                                     name="POOL3_1")

        with tf.name_scope("conv_layer_4") as scope:
            with tf.name_scope("weights") as scope:
                w4_1 = tf.Variable(tf.truncated_normal([4, 4, 128, 256], stddev=stddev_), name="w4_1")
            with tf.name_scope("biases") as scope:
                b4_1 = tf.Variable(tf.constant(0.1, shape=[256]), name="b4_1")

            conv4_1 = tf.nn.conv2d(input=pool3_1,
                                   filter=w4_1,
                                   strides=[1, 1, 1, 1],
                                   padding="SAME")
            conv4_1 += b4_1
            conv4_1 = tf.nn.relu(conv4_1, name="CONV4_1")

        with tf.name_scope("flatten_layer_1") as scope:
            shape = conv4_1.shape
            size = int(shape[1] * shape[2] * shape[3])
            flatten1_1 = tf.reshape(conv4_1, shape=[-1, size], name="FLATTEN1_1")

        with tf.name_scope("fully_layer_1") as scope:
            with tf.name_scope("weights") as scope:
                w_flat1_1 = tf.Variable(tf.truncated_normal([size, 4096], stddev=stddev_), name="w_flat1_1")
            with tf.name_scope("biases") as scope:
                b_flat1_1 = tf.Variable(tf.constant(0.1, shape=[4096]), name="b_flat1_1")

            fully1_1 = tf.matmul(flatten1_1, w_flat1_1) + b_flat1_1
            fully1_1 = tf.nn.sigmoid(fully1_1, name="FULLY1_1")

        return fully1_1


def main():
    model = siamese()

    root_path = os.path.dirname(os.path.dirname(__file__))
    siamese_path = os.path.join(root_path, "siamese_tf")
    siamese_log_dir = os.path.join(siamese_path, "log")
    if os.path.isdir(siamese_log_dir) == False:
        os.mkdir(siamese_log_dir)

    with tf.Session() as session:
        img1 = tf.placeholder(tf.float32, shape=[None, 105, 105, 3], name="img1")
        img2 = tf.placeholder(tf.float32, shape=[None, 105, 105, 3], name="img2")

        model = model.make_model(img1, img2)

        summary_writer = tf.summary.FileWriter(siamese_log_dir, session.graph)


if __name__ == '__main__':
    main()
