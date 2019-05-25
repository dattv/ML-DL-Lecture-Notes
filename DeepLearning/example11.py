import os
import tarfile

from tensorflow.python import pywrap_tensorflow
import tensorflow as tf


from tqdm import tqdm
import urllib.request


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


class VGG:
    def __init__(self, VGG_URL="http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz",
                 ):
        self._vgg_url = VGG_URL
        self._vgg_file_name = self._vgg_url.split("/")[-1]
        temp_vgg_file_name = self._vgg_file_name
        temp_vgg_file_name = temp_vgg_file_name.split(".")[0]

        self._work_dir = os.getcwd()
        temp_vgg_file_name = os.path.join(self._work_dir, temp_vgg_file_name)
        if os.path.isdir(temp_vgg_file_name) == False:
            os.mkdir(temp_vgg_file_name)

        self._vgg_file_name = os.path.join(temp_vgg_file_name, self._vgg_file_name)

        # Download VGG model from the internet
        if not os.path.exists(self._vgg_file_name):
            with tqdm(unit="B", unit_scale=True, leave=True, miniters=1, desc=VGG_URL.split("/")[-1]) as t:
                self._vgg_file_name, _ = urllib.request.urlretrieve(self._vgg_url, filename=self._vgg_file_name,
                                                                    reporthook=my_hook(t), data=None)

        # Extract the the downloaded file
        if self._vgg_file_name.endswith("gz") == True:
            with tarfile.open(name=self._vgg_file_name) as tar:
                self._vgg_model_file_name = os.path.join(temp_vgg_file_name, tar.getnames()[0])

                for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                    tar.extract(member=member, path=temp_vgg_file_name)


        # Load VGG parameters
        self.read_VGG_model(file_path=self._vgg_model_file_name)

        

    def read_VGG_model(self, file_path="", TRAINABLE=False):
        if file_path.endswith("ckpt"):

            reader = pywrap_tensorflow.NewCheckpointReader(file_path)
            var_to_shape_map = reader.get_variable_to_shape_map()

            for name in var_to_shape_map:
                print("tensor: {:30}, size: {}".format(name, var_to_shape_map[name]))

            # Layer 1
            self._weights_1_1 = reader.get_tensor("vgg_16/conv1/conv1_1/weights")
            self._t_weights_1_1 = tf.Variable(initial_value=self._weights_1_1, name='weights', trainable=TRAINABLE)

            self._bias_1 = reader.get_tensor("vgg_16/conv1/conv1_1/biases")
            self._t_bias_1_1 = tf.Variable(initial_value=self._bias_1, name='biases', trainable=TRAINABLE)

            self._weights_1_2 = reader.get_tensor("vgg_16/conv1/conv1_2/weights")
            self._t_weights_1_2 = tf.Variable(initial_value=self._weights_1_2, name='weights', trainable=TRAINABLE)

            self._bias_1_2 = reader.get_tensor("vgg_16/conv1/conv1_2/biases")
            self._t_biases_1_2 = tf.Variable(initial_value=self._bias_1_2, name='biases', trainable=TRAINABLE)

            # Layer 2
            self._weights_2_1 = reader.get_tensor("vgg_16/conv2/conv2_1/weights")
            self._t_weights_2_1 = tf.Variable(initial_value=self._weights_2_1, name='weights', trainable=TRAINABLE)

            self._biases_2_1 = reader.get_tensor("vgg_16/conv2/conv2_1/biases")
            self._t_biases_2_1 = tf.Variable(initial_value=self._biases_2_1, name='biases', trainable=TRAINABLE)

            self._weights_2_2 = reader.get_tensor("vgg_16/conv2/conv2_2/weights")
            self._t_weights_2_2 = tf.Variable(initial_value=self._weights_2_2, name='weights', trainable=TRAINABLE)

            self._biases_2_2 = reader.get_tensor("vgg_16/conv2/conv2_2/biases")
            self._t_biases_2_2 = tf.Variable(initial_value=self._biases_2_2, name='biases', trainable=TRAINABLE)

            # Layer 3
            self._weights_3_1 = reader.get_tensor("vgg_16/conv3/conv3_1/weights")
            self._t_weights_3_1 = tf.Variable(initial_value=self._weights_3_1, name='weights', trainable=TRAINABLE)

            self._biases_3_1 = reader.get_tensor("vgg_16/conv3/conv3_1/biases")
            self._t_biases_3_1 = tf.Variable(initial_value=self._biases_3_1, name='biases', trainable=TRAINABLE)

            self._weights_3_2 = reader.get_tensor("vgg_16/conv3/conv3_2/weights")
            self._t_weights_3_2 = tf.Variable(initial_value=self._weights_3_2, name='weights', trainable=TRAINABLE)

            self._biases_3_2 = reader.get_tensor("vgg_16/conv3/conv3_2/biases")
            self._t_biases_3_2 = tf.Variable(initial_value=self._biases_3_2, name='biases', trainable=TRAINABLE)

            self._weights_3_3 = reader.get_tensor("vgg_16/conv3/conv3_3/weights")
            self._t_weights_3_3 = tf.Variable(initial_value=self._weights_3_3, name='weights', trainable=TRAINABLE)

            self._biases_3_3 = reader.get_tensor("vgg_16/conv3/conv3_3/biases")
            self._t_biases_3_3 = tf.Variable(initial_value=self._biases_3_3, name='biases', trainable=TRAINABLE)

            # Layer 4
            self._weights_4_1 = reader.get_tensor("vgg_16/conv4/conv4_1/weights")
            self._t_weights_4_1 = tf.Variable(initial_value=self._weights_4_1, name='weights', trainable=TRAINABLE)

            self._biases_4_1 = reader.get_tensor("vgg_16/conv4/conv4_1/biases")
            self._t_biases_4_1 = tf.Variable(initial_value=self._biases_4_1, name='biases', trainable=TRAINABLE)

            self._weights_4_2 = reader.get_tensor("vgg_16/conv4/conv4_2/weights")
            self._t_weights_4_2 = tf.Variable(initial_value=self._weights_4_2, name='weights', trainable=TRAINABLE)

            self._biases_4_2 = reader.get_tensor("vgg_16/conv4/conv4_2/biases")
            self._t_biases_4_2 = tf.Variable(initial_value=self._biases_4_2, name='biases', trainable=TRAINABLE)

            self._weights_4_3 = reader.get_tensor("vgg_16/conv4/conv4_3/weights")
            self._t_weights_4_3 = tf.Variable(initial_value=self._weights_4_3, name='weights', trainable=TRAINABLE)

            self._biases_4_3 = reader.get_tensor("vgg_16/conv4/conv4_3/biases")
            self._t_biases_4_3 = tf.Variable(initial_value=self._biases_4_3, name='biases', trainable=TRAINABLE)

            # Layer 5
            self._weights_5_1 = reader.get_tensor("vgg_16/conv5/conv5_1/weights")
            self._t_weights_5_1 = tf.Variable(initial_value=self._weights_5_1, name='weights', trainable=TRAINABLE)

            self._biases_5_1 = reader.get_tensor("vgg_16/conv5/conv5_1/biases")
            self._t_biases_5_1 = tf.Variable(initial_value=self._biases_5_1, name='biases', trainable=TRAINABLE)

            self._weights_5_2 = reader.get_tensor("vgg_16/conv5/conv5_2/weights")
            self._t_weights_5_2 = tf.Variable(initial_value=self._weights_5_2, name='weights', trainable=TRAINABLE)

            self._biases_5_2 = reader.get_tensor("vgg_16/conv5/conv5_2/biases")
            self._t_biases_5_2 = tf.Variable(initial_value=self._biases_5_2, name='biases', trainable=TRAINABLE)

            self._weights_5_3 = reader.get_tensor("vgg_16/conv5/conv5_3/weights")
            self._t_weights_5_3 = tf.Variable(initial_value=self._weights_5_3, name='weights', trainable=TRAINABLE)

            self._biases_5_3 = reader.get_tensor("vgg_16/conv5/conv5_3/biases")
            self._t_biases_5_3 = tf.Variable(initial_value=self._biases_5_3, name='biases', trainable=TRAINABLE)

            # Layer 6
            self._weights_6 = reader.get_tensor("vgg_16/fc6/weights")
            self._t_weights_6 = tf.Variable(initial_value=self._weights_6, name='weights', trainable=TRAINABLE)

            self._biases_6 = reader.get_tensor("vgg_16/fc6/biases")
            self._t_biases_6 = tf.Variable(initial_value=self._biases_6, name='biases', trainable=TRAINABLE)

            # Layer 7
            self._weights_7 = reader.get_tensor("vgg_16/fc7/weights")
            self._t_weights_7 = tf.Variable(initial_value=self._weights_7, name='weights', trainable=TRAINABLE)

            self._biases_7 = reader.get_tensor("vgg_16/fc7/biases")
            self._t_biases_7 = tf.Variable(initial_value=self._biases_7, name='biases', trainable=TRAINABLE)

            # Layer 8
            self._weights_8 = reader.get_tensor("vgg_16/fc8/weights")
            self._t_weights_8 = tf.Variable(initial_value=self._weights_8, name='weights', trainable=True)

            self._biases_8 = reader.get_tensor("vgg_16/fc8/biases")
            self._t_biases_8 = tf.Variable(initial_value=self._biases_8, name='biases', trainable=True)


    def build_vgg(self, x_input, keep_prob=1.):
        with tf.name_scope("vgg_16") as scope:
            with tf.name_scope("conv1") as scope:
                with tf.name_scope("conv1_1") as scope:
                    self._conv1_1 = tf.nn.conv2d(input=x_input,
                                                 filter=self._t_weights_1_1,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv1_1 = self._conv1_1 + self._t_bias_1_1
                    self._conv1_1 = tf.nn.relu(self._conv1_1, name='activation')

                with tf.name_scope("conv1_2") as scope:
                    self._conv1_2 = tf.nn.conv2d(input=self._conv1_1,
                                                 filter=self._t_weights_1_2,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv1_2 = self._conv1_2 + self._t_biases_1_2
                    self._conv1_2 = tf.nn.relu(self._conv1_2, name='activation')

            with tf.name_scope("pool1") as scope:
                self._pooling1 = tf.nn.max_pool(self._conv1_2,
                                                ksize=[1, 2, 2, 1],
                                                strides=[1, 2, 2, 1],
                                                padding='VALID',
                                                name='pooling1')

            with tf.name_scope("conv2") as scope:
                with tf.name_scope("conv2_1") as scope:
                    self._conv2_1 = tf.nn.conv2d(input=self._pooling1,
                                                 filter=self._t_weights_2_1,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv2_1 = self._conv2_1 + self._t_biases_2_1
                    self._conv2_1 = tf.nn.relu(self._conv2_1)

                with tf.name_scope("conv2_2") as scope:
                    self._conv2_2 = tf.nn.conv2d(input=self._conv2_1,
                                                 filter=self._t_weights_2_2,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv2_2 = self._conv2_2 + self._t_biases_2_2
                    self._conv2_2 = tf.nn.relu(self._conv2_2, name='conv2_2')

            with tf.name_scope("pool2") as scope:
                self._pooling2 = tf.nn.max_pool(self._conv2_2,
                                                ksize=[1, 2, 2, 1],
                                                strides=[1, 2, 2, 1],
                                                padding='VALID',
                                                name='pooling2')

            with tf.name_scope("conv3") as scope:
                with tf.name_scope("conv3_1") as scope:
                    self._conv3_1 = tf.nn.conv2d(input=self._pooling2,
                                                 filter=self._t_weights_3_1,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv3_1 = self._conv3_1 + self._t_biases_3_1
                    self._conv3_1 = tf.nn.relu(self._conv3_1, name='conv3_1')

                with tf.name_scope("conv3_2") as scope:
                    self._conv3_2 = tf.nn.conv2d(input=self._conv3_1,
                                                 filter=self._t_weights_3_2,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv3_2 = self._conv3_2 + self._t_biases_3_2
                    self._conv3_2 = tf.nn.relu(self._conv3_2, name='conv3_2')

                with tf.name_scope("conv3_3") as scope:
                    self._conv3_3 = tf.nn.conv2d(input=self._conv3_2,
                                                 filter=self._t_weights_3_3,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv3_3 = self._conv3_3 + self._t_biases_3_3
                    self._conv3_3 = tf.nn.relu(self._conv3_3, name='conv3_3')

            with tf.name_scope("pool3") as scope:
                self._pooling3 = tf.nn.max_pool(self._conv3_3,
                                                ksize=[1, 2, 2, 1],
                                                strides=[1, 2, 2, 1],
                                                padding='VALID',
                                                name='pooling3')

            with tf.name_scope("conv4") as scope:
                with tf.name_scope("conv4_1") as scope:
                    self._conv4_1 = tf.nn.conv2d(input=self._pooling3,
                                                 filter=self._t_weights_4_1,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv4_1 = self._conv4_1 + self._t_biases_4_1
                    self._conv4_1 = tf.nn.relu(self._conv4_1, name='conv4_1')

                with tf.name_scope("conv4_2") as scope:
                    self._conv4_2 = tf.nn.conv2d(input=self._conv4_1,
                                                 filter=self._t_weights_4_2,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv4_2 = self._conv4_2 + self._t_biases_4_2
                    self._conv4_2 = tf.nn.relu(self._conv4_2, name='conv4_2')

                with tf.name_scope("conv4_3") as scope:
                    self._conv4_3 = tf.nn.conv2d(input=self._conv4_2,
                                                 filter=self._t_weights_4_3,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv4_3 = self._conv4_3 + self._t_biases_4_3
                    self._conv4_3 = tf.nn.relu(self._conv4_3, name='conv4_3')

            with tf.name_scope("pool4") as scope:
                self._pooling4 = tf.nn.max_pool(self._conv4_3,
                                                ksize=[1, 2, 2, 1],
                                                strides=[1, 2, 2, 1],
                                                padding='VALID',
                                                name='pooling4')

            with tf.name_scope("conv5") as scope:
                with tf.name_scope("conv5_1") as scope:
                    self._conv5_1 = tf.nn.conv2d(input=self._pooling4,
                                                 filter=self._t_weights_5_1,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv5_1 = self._conv5_1 + self._t_biases_5_1
                    self._conv5_1 = tf.nn.relu(self._conv5_1, name='conv5_1')

                with tf.name_scope("conv5_2") as scope:
                    self._conv5_2 = tf.nn.conv2d(input=self._conv5_1,
                                                 filter=self._t_weights_5_2,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv5_2 = self._conv5_2 + self._t_biases_5_2
                    self._conv5_2 = tf.nn.relu(self._conv5_2, name='conv5_2')

                with tf.name_scope("conv5_3") as scope:
                    self._conv5_3 = tf.nn.conv2d(input=self._conv5_2,
                                                 filter=self._t_weights_5_3,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv5_3 = self._conv5_3 + self._t_biases_5_3
                    self._conv5_3 = tf.nn.relu(self._conv5_3, name='conv5_3')

            with tf.name_scope("pool5") as scope:
                self._pooling5 = tf.nn.max_pool(self._conv5_3,
                                                ksize=[1, 2, 2, 1],
                                                strides=[1, 2, 2, 1],
                                                padding='VALID',
                                                name='pooling5')

            with tf.name_scope("dense6") as scope:
                self._fc6 = tf.nn.conv2d(input=self._pooling5,
                                         filter=self._t_weights_6,
                                         strides=[1, 1, 1, 1],
                                         padding='VALID')
                self._fc6 = self._fc6 + self._t_biases_6

                self._fc6 = tf.nn.relu(self._fc6, name='fc6')

                self._fc6 = tf.nn.dropout(self._fc6, keep_prob=keep_prob)

            with tf.name_scope("dense7") as scope:
                self._fc7 = tf.nn.conv2d(input=self._fc6,
                                         filter=self._t_weights_7,
                                         strides=[1, 1, 1, 1],
                                         padding='VALID')

                self._fc7 = self._fc7 + self._t_biases_7
                self._fc7 = tf.nn.relu(self._fc7, name='fc7')

                self._fc7 = tf.nn.dropout(self._fc7, keep_prob=keep_prob)

            with tf.name_scope("dense8") as scope:
                self._fc8 = tf.nn.conv2d(input=self._fc7,
                                         filter=self._t_weights_8,
                                         strides=[1, 1, 1, 1],
                                         padding='VALID')
                self._fc8 = self._fc8 + self._t_biases_8
                self._fc8 = tf.nn.softmax(self._fc8, name='fc8')

    def build_VGG_classify(self, x_input, keep_prob=1, n_output=1000):
        with tf.name_scope("vgg_16") as scope:
            with tf.name_scope("conv1") as scope:
                with tf.name_scope("conv1_1") as scope:
                    self._conv1_1 = tf.nn.conv2d(input=x_input,
                                                 filter=self._t_weights_1_1,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv1_1 = self._conv1_1 + self._t_bias_1_1
                    self._conv1_1 = tf.nn.relu(self._conv1_1, name='activation')

                with tf.name_scope("conv1_2") as scope:
                    self._conv1_2 = tf.nn.conv2d(input=self._conv1_1,
                                                 filter=self._t_weights_1_2,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv1_2 = self._conv1_2 + self._t_biases_1_2
                    self._conv1_2 = tf.nn.relu(self._conv1_2, name='activation')

            with tf.name_scope("pool1") as scope:
                self._pooling1 = tf.nn.max_pool(self._conv1_2,
                                                ksize=[1, 2, 2, 1],
                                                strides=[1, 2, 2, 1],
                                                padding='VALID',
                                                name='pooling1')

            with tf.name_scope("conv2") as scope:
                with tf.name_scope("conv2_1") as scope:
                    self._conv2_1 = tf.nn.conv2d(input=self._pooling1,
                                                 filter=self._t_weights_2_1,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv2_1 = self._conv2_1 + self._t_biases_2_1
                    self._conv2_1 = tf.nn.relu(self._conv2_1)

                with tf.name_scope("conv2_2") as scope:
                    self._conv2_2 = tf.nn.conv2d(input=self._conv2_1,
                                                 filter=self._t_weights_2_2,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv2_2 = self._conv2_2 + self._t_biases_2_2
                    self._conv2_2 = tf.nn.relu(self._conv2_2, name='conv2_2')

            with tf.name_scope("pool2") as scope:
                self._pooling2 = tf.nn.max_pool(self._conv2_2,
                                                ksize=[1, 2, 2, 1],
                                                strides=[1, 2, 2, 1],
                                                padding='VALID',
                                                name='pooling2')

            with tf.name_scope("conv3") as scope:
                with tf.name_scope("conv3_1") as scope:
                    self._conv3_1 = tf.nn.conv2d(input=self._pooling2,
                                                 filter=self._t_weights_3_1,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv3_1 = self._conv3_1 + self._t_biases_3_1
                    self._conv3_1 = tf.nn.relu(self._conv3_1, name='conv3_1')

                with tf.name_scope("conv3_2") as scope:
                    self._conv3_2 = tf.nn.conv2d(input=self._conv3_1,
                                                 filter=self._t_weights_3_2,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv3_2 = self._conv3_2 + self._t_biases_3_2
                    self._conv3_2 = tf.nn.relu(self._conv3_2, name='conv3_2')

                with tf.name_scope("conv3_3") as scope:
                    self._conv3_3 = tf.nn.conv2d(input=self._conv3_2,
                                                 filter=self._t_weights_3_3,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv3_3 = self._conv3_3 + self._t_biases_3_3
                    self._conv3_3 = tf.nn.relu(self._conv3_3, name='conv3_3')

            with tf.name_scope("pool3") as scope:
                self._pooling3 = tf.nn.max_pool(self._conv3_3,
                                                ksize=[1, 2, 2, 1],
                                                strides=[1, 2, 2, 1],
                                                padding='VALID',
                                                name='pooling3')

            with tf.name_scope("conv4") as scope:
                with tf.name_scope("conv4_1") as scope:
                    self._conv4_1 = tf.nn.conv2d(input=self._pooling3,
                                                 filter=self._t_weights_4_1,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv4_1 = self._conv4_1 + self._t_biases_4_1
                    self._conv4_1 = tf.nn.relu(self._conv4_1, name='conv4_1')

                with tf.name_scope("conv4_2") as scope:
                    self._conv4_2 = tf.nn.conv2d(input=self._conv4_1,
                                                 filter=self._t_weights_4_2,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv4_2 = self._conv4_2 + self._t_biases_4_2
                    self._conv4_2 = tf.nn.relu(self._conv4_2, name='conv4_2')

                with tf.name_scope("conv4_3") as scope:
                    self._conv4_3 = tf.nn.conv2d(input=self._conv4_2,
                                                 filter=self._t_weights_4_3,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv4_3 = self._conv4_3 + self._t_biases_4_3
                    self._conv4_3 = tf.nn.relu(self._conv4_3, name='conv4_3')

            with tf.name_scope("pool4") as scope:
                self._pooling4 = tf.nn.max_pool(self._conv4_3,
                                                ksize=[1, 2, 2, 1],
                                                strides=[1, 2, 2, 1],
                                                padding='VALID',
                                                name='pooling4')

            with tf.name_scope("conv5") as scope:
                with tf.name_scope("conv5_1") as scope:
                    self._conv5_1 = tf.nn.conv2d(input=self._pooling4,
                                                 filter=self._t_weights_5_1,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv5_1 = self._conv5_1 + self._t_biases_5_1
                    self._conv5_1 = tf.nn.relu(self._conv5_1, name='conv5_1')

                with tf.name_scope("conv5_2") as scope:
                    self._conv5_2 = tf.nn.conv2d(input=self._conv5_1,
                                                 filter=self._t_weights_5_2,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv5_2 = self._conv5_2 + self._t_biases_5_2
                    self._conv5_2 = tf.nn.relu(self._conv5_2, name='conv5_2')

                with tf.name_scope("conv5_3") as scope:
                    self._conv5_3 = tf.nn.conv2d(input=self._conv5_2,
                                                 filter=self._t_weights_5_3,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv5_3 = self._conv5_3 + self._t_biases_5_3
                    self._conv5_3 = tf.nn.relu(self._conv5_3, name='conv5_3')

            with tf.name_scope("pool5") as scope:
                self._pooling5 = tf.nn.max_pool(self._conv5_3,
                                                ksize=[1, 2, 2, 1],
                                                strides=[1, 2, 2, 1],
                                                padding='VALID',
                                                name='pooling5')

            with tf.name_scope("dense6") as scope:
                self._fc6 = tf.nn.conv2d(input=self._pooling5,
                                         filter=self._t_weights_6,
                                         strides=[1, 1, 1, 1],
                                         padding='VALID')
                self._fc6 = self._fc6 + self._t_biases_6

                self._fc6 = tf.nn.relu(self._fc6, name='fc6')

                self._fc6 = tf.nn.dropout(self._fc6, keep_prob=keep_prob)

            with tf.name_scope("dense7") as scope:
                self._fc7 = tf.nn.conv2d(input=self._fc6,
                                         filter=self._t_weights_7,
                                         strides=[1, 1, 1, 1],
                                         padding='VALID')

                self._fc7 = self._fc7 + self._t_biases_7
                self._fc7 = tf.nn.relu(self._fc7, name='fc7')

                self._fc7 = tf.nn.dropout(self._fc7, keep_prob=keep_prob)

            with tf.name_scope("vgg_16") as scope:
                with tf.name_scope("fc8") as scope:
                    with tf.name_scope("new_weights") as scope:
                        self._new_t_weights_8 = tf.Variable(tf.random_normal([1, 1, 4096, int(n_output)],
                                                                             stddev=0.1), name="new_t_weights_8",
                                                            trainable=True)
                    with tf.name_scope("new_biases") as scope:
                        self._new_t_biases_8 = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[n_output]),
                                                           name="new_t_biases_8",
                                                           trainable=True)


            with tf.name_scope("dense8") as scope:
                self._fc8 = tf.nn.conv2d(input=self._fc7,
                                         filter=self._new_t_weights_8,
                                         strides=[1, 1, 1, 1],
                                         padding='VALID')
                self._fc8 = self._fc8 + self._new_t_biases_8
                self._fc8 = tf.nn.softmax(self._fc8, name='fc8')

        return self._fc8



VGG16 = VGG()
n_output = 10
x_input = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='x_input')
y_input = tf.placeholder(dtype=tf.float32, shape=[None, n_output], name='y_input')
logits = VGG16.build_VGG_classify(x_input, keep_prob=0.5, n_output=n_output)


