import tensorflow as tf

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

z = x + y

b = tf.placeholder(tf.int8, shape=(), name="b")
i = tf.cast(b, tf.bool)

session = tf.Session()

values = {x: 0.5, y: 4.0}

result = session.run([z], values)
print(result)

result2 = session.run([i], feed_dict={b: 0})
print("convert: ", result2)


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

