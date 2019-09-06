import tensorflow as tf
# from DeepLearning.mnist_batch_norm import build_graph


def build_graph(x, is_training, create_training_graph, graph):
    # Placeholders
    # x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    # y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_")

    x_reshape = tf.reshape(x, shape=[-1, 28, 28, 1], name="x_reshape")

    # Layer 1
    z1 = tf.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid')(x_reshape)
    # bn1 = tf.layers.batch_normalization(z1, training=is_training, fused=False)
    bn1 = tf.keras.layers.BatchNormalization(fused=False)(z1)
    l1 = tf.nn.relu6(bn1)

    l1 = tf.layers.max_pooling2d(l1, pool_size=(2, 2), strides=(2, 2))

    # Layer 2
    z2 = tf.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid')(l1)
    # bn2 = tf.layers.batch_normalization(z2, training=is_training, fused=False)
    bn2 = tf.keras.layers.BatchNormalization(fused=False)(z2)
    l2 = tf.nn.relu6(bn2)

    l2 = tf.layers.max_pooling2d(l2, pool_size=(2, 2), strides=(2, 2))

    l2 = tf.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="valid")(l2)
    # l2 = tf.layers.batch_normalization(l2, training=is_training, fused=False)
    l2 = tf.keras.layers.BatchNormalization(fused=False)(l2)
    l2 = tf.nn.relu6(l2)

    # Softmax
    b3 = tf.layers.flatten(l2)
    y = tf.layers.dense(b3, units=10, activation=tf.nn.softmax, name="y")

    # # Loss, Optimizer and Predictions
    # cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    #
    # if create_training_graph is True:
    #     tf.contrib.quantize.create_training_graph(input_graph=graph)
    # else:
    #     tf.contrib.quantize.create_eval_graph(input_graph=graph)
    #
    # update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    #     train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    #
    # correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # return (x, y_), train_step, accuracy, y, tf.train.Saver()

    return y


tf.reset_default_graph()
# eval_graph = tf.get_default_graph()
# (x, y_), _, accuracy, y, saver = build_graph(is_training=False, create_training_graph=False, graph=eval_graph)
x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
y = build_graph(x, is_training=False, create_training_graph=False, graph=None)

tf.contrib.quantize.create_eval_graph()

saver = tf.train.Saver()
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    saver.restore(session, './temp-bn-save/cifar.ckpt')

    eval_graph_file = "./eval_graph_def.pb"
    checkpoint_name = "./checkpoint/checkpoint.ckpt"

    # Save GraphDef
    tf.train.write_graph(session.graph_def,'.','graph.pb')
    # Save checkpoint
    saver.save(sess=session, save_path=checkpoint_name)


    with open(eval_graph_file, "w") as f:
        f.write(str(session.graph.as_graph_def()))

    saver.save(session, checkpoint_name)

    # builder = tf.saved_model.Builder('exports')
    #
    # signature_def = tf.saved_model.predict_signature_def(
    #     inputs={'x': x},
    #     outputs={'y/Softmax': y}
    # )
    #
    # builder.add_meta_graph_and_variables(
    #     sess=session,
    #     tags=[
    #         tf.saved_model.tag_constants.SERVING
    #     ],
    #     signature_def_map={
    #         tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
    #     },
    #     saver=saver
    # )
    #
    # builder.save()

    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        session,
        session.graph_def,
        ["y/Softmax"]
    )

    with open('abc_frozen_model.pb', 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())