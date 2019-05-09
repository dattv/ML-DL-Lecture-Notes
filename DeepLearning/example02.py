import tensorflow as tf
import os

LOGDIR = "./tmp/"

fileScript = os.path.basename(__file__)
log_name = fileScript.split(".")
log_name = log_name[0]

if os.path.isdir(LOGDIR) == False:
    os.mkdir(LOGDIR)

x = tf.placeholder(tf.float32, name='x')
y = tf.placeholder(tf.float32, name='y')

z = tf.add(x, y, name='sum')

session = tf.Session()

summary_writer = tf.summary.FileWriter(os.path.join(LOGDIR, log_name), session.graph)
