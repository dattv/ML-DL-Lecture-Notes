import tensorflow as tf

class ResNet34:
    def __init__(self, input_tensor, output_class, layer_num):
        print("ResNet")
        self._dropout_probability = 0
        self._weight_decay = 0.0001
        self._momentum = 0.9
        self._use_bias = False
        self._weight_init = tf.contrib.layers.xavier_initializer(uniform=False)  # tf.initializers.he_normal()

        self._input = input_tensor

        self._layer_num = layer_num

    def ResNet34_basic(self, n_input_chanel, n_output_chanel, stride, n_sub_layer):
        def f(net):
            convs = tf.layers.conv2d(inputs=net, filters=64, kernel_size=3, strides=(1, 1), padding="same")




if __name__ == '__main__':
    net = ResNet()