import tensorflow as tf

# make and save a simple graph
G = tf.Graph()
with G.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=(), name="x")
    a = tf.Variable(5.0, name="a")
    y = tf.add(a, x, name="y")
    saver = tf.train.Saver()

with tf.Session(graph=G) as sess:
    sess.run(tf.global_variables_initializer())
    out = sess.run(fetches=[y], feed_dict={x: 1.0})

    # Save GraphDef
    tf.train.write_graph(sess.graph_def,'.','graph.pb')
    # Save checkpoint
    saver.save(sess=sess, save_path="test_model")

