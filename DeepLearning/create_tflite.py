import tensorflow.contrib.lite as lite
import tensorflow as tf

# converter = lite.TFLiteConverter.from_saved_model('/home/dat/PycharmProjects/ML-DL-Lecture-Notes/DeepLearning/exports')
converter = tf.lite.TFLiteConverter.from_frozen_graph(
    "/home/dat/PycharmProjects/ML-DL-Lecture-Notes/DeepLearning/optimized_model.pb",
    input_arrays=["x"],
    output_arrays=["y/Softmax"],
    input_shapes={'x': [1, 784]})

input_arrays = converter.get_input_arrays()
print(input_arrays)
converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
converter.inference_input_type = tf.lite.constants.QUANTIZED_UINT8
converter.quantized_input_stats = {'x': (0.0, 255.0)}

converter.dump_graphviz_dir = './'

flatbuffer = converter.convert()

with open('mnist.tflite', 'wb') as outfile:
    outfile.write(flatbuffer)
