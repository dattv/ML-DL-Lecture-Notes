import tensorflow as tf



graph_def_file = 'frozen_model.pb'
inputs = ['x_input']
outputs = ['WideResNet/classifier_block/predict_gender/softmax']

converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, inputs, outputs)
converter.inference_type = tf.lite.constants.QUANTIZED_UINT8      ## Indicates TPU compatibility
input_arrays = converter.get_input_arrays()
print(input_arrays)
converter.quantized_input_stats = {input_arrays[0]: (0., 1.)}     ## mean, std_dev
converter.default_ranges_stats = (-128, 127)                      ## min, max values for quantization (?)
# converter.allow_custom_ops = True                                 ## not sure if this is needed

## REMOVED THE OPTIMIZATIONS ALTOGETHER TO MAKE IT WORK

tflite_model = converter.convert()

open('model.tflite', 'wb').write(tflite_model)