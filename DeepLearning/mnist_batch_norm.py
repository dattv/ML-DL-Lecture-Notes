import numpy
import numpy as np, tensorflow as tf, tqdm
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Generate predetermined random weights so the networks are similarly initialized
w1_initial = np.random.normal(size=(784, 100)).astype(np.float32)
w2_initial = np.random.normal(size=(100, 100)).astype(np.float32)
w3_initial = np.random.normal(size=(100, 10)).astype(np.float32)

# Small epsilon value for the BN transform
epsilon = 1e-3

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


def build_graph(x):
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

    return y

# Build training graph, train and save the trained model
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
y = build_graph(x,)
y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_")
# Loss, Optimizer and Predictions
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))


tf.contrib.quantize.create_training_graph()

update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
acc = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("./log_mnist_batch_norm/", sess.graph)

    for i in tqdm.tqdm(range(100)):
        batch = mnist.train.next_batch(16)
        sess.run([train_step], feed_dict={x: batch[0], y_: batch[1]})
        if i % 50 is 0:
            res = sess.run([accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            acc.append(res[0])

    saved_model = saver.save(sess, './temp-bn-save/cifar.ckpt')

print("Final accuracy:", acc[-1])

# ======================================================================================================================
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
y = build_graph(x)
y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_")
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
predictions = []
correct = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './temp-bn-save/cifar.ckpt')
    for i in range(100):
        pred, corr = sess.run([tf.arg_max(y, 1), accuracy],
                              feed_dict={x: [mnist.test.images[i]], y_: [mnist.test.labels[i]]})
        correct += corr
        predictions.append(pred[0])
print("PREDICTIONS:", predictions)
print("EXACT RES  :", [numpy.argmax(f) for f in mnist.test.labels[0:100]])
print("ACCURACY:", correct / 100)

# ======================================================================================================================
#
# tf.reset_default_graph()
# eval_graph = tf.get_default_graph()
# # (x, y_), _, accuracy, y, saver = build_graph(is_training=False, create_training_graph=False, graph=eval_graph)
# x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
# y = build_graph(x, is_training=False, create_training_graph=False, graph=eval_graph)
#
# tf.contrib.quantize.create_eval_graph(input_graph=eval_graph)
#
# saver = tf.train.Saver()
# with tf.Session(graph=eval_graph) as session:
#     session.run(tf.global_variables_initializer())
#     saver.restore(session, './temp-bn-save/cifar.ckpt')
#
#     eval_graph_file = "./eval_graph_def.pb"
#     checkpoint_name = "./checkpoint/checkpoint.ckpt"
#
#     # Save GraphDef
#     tf.train.write_graph(session.graph_def,'.','graph.pb')
#     # Save checkpoint
#     saver.save(sess=session, save_path=checkpoint_name)
#
#
#     with open(eval_graph_file, "w") as f:
#         f.write(str(session.graph.as_graph_def()))
#
#     saver.save(session, checkpoint_name)
#
#     # builder = tf.saved_model.Builder('exports')
#     #
#     # signature_def = tf.saved_model.predict_signature_def(
#     #     inputs={'x': x},
#     #     outputs={'y/Softmax': y}
#     # )
#     #
#     # builder.add_meta_graph_and_variables(
#     #     sess=session,
#     #     tags=[
#     #         tf.saved_model.tag_constants.SERVING
#     #     ],
#     #     signature_def_map={
#     #         tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
#     #     },
#     #     saver=saver
#     # )
#     #
#     # builder.save()
#
#     frozen_graph_def = tf.graph_util.convert_variables_to_constants(
#         session,
#         sess.graph_def,
#         ["y/Softmax"]
#     )
#
#     with open('abc_frozen_model.pb', 'wb') as f:
#         f.write(frozen_graph_def.SerializeToString())
#
# # ======================================================================================================================
# from tensorflow.python.tools.freeze_graph import freeze_graph
#
# input_saver_def_path = ""
# input_binary = False
# checkpoint_path = checkpoint_name
# output_node_names = "y/Softmax"
# restore_op_name = "save/restore_all"
# filename_tensor_name = "save/Const:0"
# output_graph_path = "./frozen_model.pb"
# clear_devices = False
# freeze_graph(eval_graph_file,
#              input_saver_def_path,
#              input_binary,
#              checkpoint_path,
#              output_node_names,
#              restore_op_name,
#              filename_tensor_name,
#              output_graph_path,
#              clear_devices, "")
#
# # ======================================================================================================================
# from tensorflow.python.tools.import_pb_to_tensorboard import import_to_tensorboard
#
# frozen_log = "./frozen_log"
# import_to_tensorboard(model_dir=output_graph_path, log_dir=frozen_log)
#
# # ======================================================================================================================
# from tensorflow.python.tools import optimize_for_inference, optimize_for_inference_lib
#
# input_graph_def = tf.GraphDef()
# with tf.gfile.Open(output_graph_path, "rb") as f:
#     data = f.read()
#     input_graph_def.ParseFromString(data)
#
# output_graph_def = optimize_for_inference_lib.optimize_for_inference(
#     input_graph_def,
#     ["x"],  ## input
#     ["y/Softmax"],  ## outputs
#     tf.float32.as_datatype_enum)
#
# f = tf.gfile.FastGFile("./optimized_model.pb", "wb")
# f.write(output_graph_def.SerializeToString())
#
