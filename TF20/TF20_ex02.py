import tensorflow as tf

var = tf.Variable([3, 3])
if tf.test.is_gpu_available():
    print("running on GPU")
    print("GPU: {}".format(var.device.endswith('GPU:0')))
else:
    print("Running on CPU")
