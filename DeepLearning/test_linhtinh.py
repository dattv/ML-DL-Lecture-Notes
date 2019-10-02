import os

import tensorflow as tf

saved_model_dir = "./TFEstimator_cifar10_bin_model/export/Servo/1568993997"
print(os.path.isdir(saved_model_dir))
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
