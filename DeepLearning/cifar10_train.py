import tensorflow as tf
import numpy as np
import os

from DN_network import classsification10_nn
from cifar_data_set import building_dataset


cifar10_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
data = building_dataset.building_cifar(cifar10_url)
data = data.data

X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="input")
TARGET = tf.placeholder(tf.float32, shape=[None, 10], name="target")
keep_prop = tf.placeholder(tf.float32, name="keep_prop")

with tf.variable_scope("ANN") as scope:
    net = classsification10_nn.classification10_nn(X, keep_prop, [30, 50, 80, 500, 10])
    output = net.output

with tf.variable_scope("loss") as scope:
    softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=output,
                                                                    labels=TARGET)

    loss_operation = tf.reduce_mean(softmax_cross_entropy, name="loss")

    tf.summary.scalar("loss", loss_operation)

with tf.variable_scope("optimization") as scope:
    optimiser = tf.train.AdamOptimizer(1.e-5).minimize(loss_operation)

with tf.variable_scope("accuracy") as scope:
    with tf.name_scope("correct_preduction") as scope:
        predictions = tf.argmax(output, 1)
        correct_prediction = tf.equal(predictions, tf.argmax(TARGET, 1))

    with tf.name_scope("acc") as scope:
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar("accuracy", accuracy_operation)

merged_summary_operation = tf.summary.merge_all()
saver = tf.train.Saver()

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    train_writer = tf.summary.FileWriter("./log/train", session.graph)
    test_writer = tf.summary.FileWriter("./log/test", session.graph)
    builder = tf.saved_model.Builder("./pb")
    builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.TRAINING])

    for epoch in range(50000):
        batch = data.train.nex_batch(500)
        _, merge_sum = session.run([optimiser, merged_summary_operation],
                                   feed_dict={X: batch[0],
                                              TARGET: batch[1],
                                              keep_prop: 0.5})

        train_writer.add_summary(merge_sum, epoch)

        if epoch % 1000 == 0:
            test_imgs = data.test.images.reshape(10, 1000, 32, 32, 3)
            test_lbs = data.test.labels.reshape(10, 1000, 10)

            acc = np.mean([session.run([accuracy_operation],
                                       feed_dict={X: test_imgs[i],
                                                  TARGET: test_lbs[i],
                                                  keep_prop: 1.}) for i in range(10)])

            print("EPOCH: {}, ACC: {}".format(epoch, acc * 100))

            test_writer.add_summary(merge_sum, epoch)

    saver.save(session, "./ckpt/ckpt_cifar10_model.ckpt")
    builder.save()