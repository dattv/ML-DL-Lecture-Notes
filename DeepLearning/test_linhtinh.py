import tensorflow as tf
slim = tf.contrib.slim
file_pattern = '/media/dat/68fa98f8-9d03-4c1e-9bdb-c71ea72ab6fa/dat/dataset/tfrecord/tfrecord_ImageNet/train-*-of-01024' #文件名格式

# 适配器1：将example反序列化成存储之前的格式。由tf完成
keys_to_features = {
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
    'image/height': tf.FixedLenFeature([1], tf.int64),
    'image/width': tf.FixedLenFeature([1], tf.int64),
    'image/channels': tf.FixedLenFeature([1], tf.int64),
    'image/shape': tf.FixedLenFeature([3], tf.int64),
    'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
    'image/class/text': tf.FixedLenFeature([], tf.string, default_value=""),
    "image/filename": tf.FixedLenFeature([], tf.string, default_value=""),
    "image/class/label": tf.FixedLenFeature([1], tf.int64, default_value=[0]),
}

#适配器2：将反序列化的数据组装成更高级的格式。由slim完成，即可以将上面的几个数据包装成一个新数据
items_to_handlers = {
    'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
    'label': slim.tfexample_decoder.Tensor('image/class/label'),
}

# 解码器
decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

# dataset对象定义了数据集的文件位置，解码方式等元信息
dataset = slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=tf.TFRecordReader,
            num_samples = 3, # 手动生成了三个文件， 每个文件里只包含一个example
            decoder=decoder,
            items_to_descriptions = {},
            num_classes=21)

#provider对象根据dataset信息读取数据
provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers=3,
                    shuffle=False)

[image, label] = provider.get(['image', 'label'])
print(type(image))
print(image.shape)
print(image)