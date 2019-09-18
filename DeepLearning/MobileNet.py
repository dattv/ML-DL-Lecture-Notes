import tensorflow as tf

l = tf.keras.layers


def _conv(inputs, filters, kernel_size, strides, padding, bias=False, normalize=True, activation='relu'):
    output = inputs
    padding_str = 'same'
    if padding > 0:
        output = tf.keras.layers.ZeroPadding2D(padding=padding)(output)
        padding_str = 'valid'
    output = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding_str, use_bias=bias,
                                    kernel_initializer='he_normal',
                                    kernel_regularizer=tf.keras.regularizers.l2(l=5e-4))(output)
    if normalize:
        output = tf.keras.layers.BatchNormalization(axis=3)(output)
    if activation == 'relu':
        output = tf.keras.layers.ReLU()(output)
    if activation == 'relu6':
        output = tf.keras.layers.ReLU(max_value=6)(output)
    if activation == 'leaky_relu':
        output = tf.keras.layers.LeakyReLU(alpha=0.1)(output)
    return output


def _dwconv(inputs, filters, kernel_size, strides, padding, bias=False, activation='relu'):
    output = inputs
    padding_str = 'same'
    if padding > 0:
        output = tf.keras.layers.ZeroPadding2D(padding=(padding, padding))(output)
        padding_str = 'valid'
    output = tf.keras.layers.DepthwiseConv2D(kernel_size, strides, padding_str, use_bias=bias,
                                             depthwise_initializer='he_uniform',
                                             depthwise_regularizer=tf.keras.regularizers.l2(5e-4))(output)
    output = tf.keras.layers.BatchNormalization(axis=3)(output)
    if activation == 'relu':
        output = tf.keras.layers.ReLU()(output)
    if activation == 'relu6':
        output = tf.keras.layers.ReLU(max_value=6)(output)
    if activation == 'leaky_relu':
        output = tf.keras.layers.LeakyReLU(alpha=0.1)(output)
    return output


def _bottleneck(inputs, in_filters, out_filters, kernel_size, strides, bias=False, activation='relu6', t=1):
    output = inputs
    output = _conv(output, in_filters * t, 1, 1, 0, False, activation)
    padding = 0
    if strides == 2:
        padding = 1
    output = _dwconv(output, in_filters * t, kernel_size, strides, padding, bias=False, activation=activation)
    output = _conv(output, out_filters, 1, 1, 0, False, 'linear')
    if strides == 1 and inputs.get_shape().as_list()[3] == output.get_shape().as_list()[3]:
        output = tf.keras.layers.add([output, inputs])
    return output


def mobilenet_model_v1(ImageHeight, ImageWidth):
    # Input Layer
    image = tf.keras.Input(shape=(ImageHeight, ImageWidth, 3))
    net = _conv(image, 32, 3, 2, 1)
    net = _dwconv(net, 32, 3, 1, 0)
    net = _conv(net, 64, 1, 1, 0)
    net = _dwconv(net, 64, 3, 2, 1)
    net = _conv(net, 128, 1, 1, 0)
    net = _dwconv(net, 128, 3, 1, 0)
    net = _conv(net, 128, 1, 1, 0)
    net = _dwconv(net, 128, 3, 2, 1)
    net = _conv(net, 256, 1, 1, 0)
    net = _dwconv(net, 256, 3, 1, 0)
    net = _conv(net, 256, 1, 1, 0)
    net = _dwconv(net, 256, 3, 2, 1)
    net = _conv(net, 512, 1, 1, 0)
    for _ in range(5):
        net = _dwconv(net, 512, 3, 1, 0)
        net = _conv(net, 512, 1, 1, 0)
    net = _dwconv(net, 512, 3, 2, 1)
    net = _conv(net, 1024, 1, 1, 0)
    net = _dwconv(net, 1024, 3, 1, 0)
    net = _conv(net, 1024, 1, 1, 0)
    net = l.GlobalAveragePooling2D()(net)
    net = l.Flatten()(net)
    logits = l.Dense(1000, kernel_initializer=tf.initializers.truncated_normal(stddev=1 / 1000))(net)
    model = tf.keras.Model(inputs=image, outputs=logits)
    return model


def mobilenet_model_v2(ImageHeight, ImageWidth):
    # Input Layer
    image = tf.keras.Input(shape=(ImageHeight, ImageWidth, 3))  # 224*224*3
    net = _conv(image, 32, 3, 2, 1, False, 'relu6')  # 112*112*32
    net = _bottleneck(net, 32, 16, 3, 1, False, 'relu6', 1)  # 112*112*16
    net = _bottleneck(net, 16, 24, 3, 2, False, 'relu6', 6)  # 56*56*24
    # net = _bottleneck(net, 24, 24, 3, 1, False, 'relu6', 6)  # 56*56*24
    net = _bottleneck(net, 24, 32, 3, 2, False, 'relu6', 6)  # 28*28*32
    # net = _bottleneck(net, 32, 32, 3, 1, False, 'relu6', 6)  # 28*28*32
    # net = _bottleneck(net, 32, 32, 3, 1, False, 'relu6', 6)  # 28*28*32
    net = _bottleneck(net, 32, 64, 3, 2, False, 'relu6', 6)  # 14*14*64
    # net = _bottleneck(net, 64, 64, 3, 1, False, 'relu6', 6)  # 14*14*64
    # net = _bottleneck(net, 64, 64, 3, 1, False, 'relu6', 6)  # 14*14*64
    # net = _bottleneck(net, 64, 64, 3, 1, False, 'relu6', 6)  # 14*14*64
    net = _bottleneck(net, 64, 96, 3, 1, False, 'relu6', 6)  # 14*14*96
    # net = _bottleneck(net, 96, 96, 3, 1, False, 'relu6', 6)  # 14*14*96
    # net = _bottleneck(net, 96, 96, 3, 1, False, 'relu6', 6)  # 14*14*96
    # net = _bottleneck(net, 96, 96, 3, 1, False, 'relu6', 6)  # 14*14*96
    net = _bottleneck(net, 96, 160, 3, 2, False, 'relu6', 6)  # 7*7*160
    # net = _bottleneck(net, 160, 160, 3, 1, False, 'relu6', 6)  # 7*7*160
    # net = _bottleneck(net, 160, 160, 3, 1, False, 'relu6', 6)  # 7*7*160
    # net = _bottleneck(net, 160, 320, 3, 1, False, 'relu6', 6)  # 7*7*320
    net = _conv(net, 1280, 3, 1, 0, False, 'relu6')  # 7*7*1280
    net = l.AveragePooling2D(7)(net)
    net = l.Flatten()(net)
    logits = l.Dense(1000, kernel_initializer=tf.initializers.truncated_normal(stddev=1 / 1000))(net)
    model = tf.keras.Model(inputs=image, outputs=logits)
    return model
