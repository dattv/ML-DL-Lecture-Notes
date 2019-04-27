The TFRecord.md is for learning Tensorflow Records. What they are? and how they can be used?

# Overview
Tensorflow is a second machine learning framwork that Google created for researching and developing AI/ML or DNN application, it is widely used in both academic and engineering society. <b>TFRecord</b> is a kind of Tensorflow's own binary format.
<br />
<br /> <b> Advantages:</b>
<br /> If you are working with large datasets, using a binary format for storaging and reading of your data can hav significan impact on the performance of your import and consequence on the training time of your model.
<br /> :+1: Binary format take up less space on your disk.
<br /> :+1: Binary format take less time to copy and can be read much more efficiently from disk.
<br /> :+1: It easy to combine multiple datasets and intergrates seamlessly with the data import and preprocessing functionality provided by library.
<br /> :+1: For dataset that are too large to be stored fully in memory this is an advantage as only the data that is require at the time (eg. batch) is loaded form disk and then processed.
<br /> :+1: It's possible to store sequence data []()

<br /> <b> Disadvantages:</b>
<br /> :-1: You have to convert your data to this format and there are quite little documents for descripting fully how to do that task.

## Methodology
A TFRecord file store your data as a sequence of binary strings. This mean that you need to desclare theis format before you can write it in to files.
<br /> Tensorflow give us two APIs to do this purpose: 
- ![tf.train.Example](https://placehold.it/15/f03c15/000000?text=+)`tf.train.Example`
- ![tf.train.Example](https://placehold.it/15/f03c15/000000?text=+)`tf.train.SequenceExample`
<br /> And then you can use:
- ![tf.train.Example](https://placehold.it/15/f03c15/000000?text=+)`tf.python_io.TFRecordWriter`
<br /> to write them to your disk.

## How to use
### `tf.train.Example`
*If you have dataset consist of feature, where each feature is a list of value of the same type, `tf.train.Example` is a right way to use*
<br /> the movie recomendation  application:


/***
| __Animals__ | __Sports__ | __Fruits__ |
|-------------|------------|------------|
| Cat         | Soccer     | Apple      |
| Dog         | Basketball | Orange     |
***/       


# Reference
[1] [Tensorflow Records? What they are and how to use them](https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564)
<br /> [2] [How to use TFRecord with Datasets and Iterators in Tensorflow with code samples](https://medium.com/ymedialabs-innovation/how-to-use-tfrecord-with-datasets-and-iterators-in-tensorflow-with-code-samples-ffee57d298af)
<br /> [3] [TensorFlow Tutorial For Beginners](https://www.datacamp.com/community/tutorials/tensorflow-tutorial?utm_source=adwords_ppc&utm_campaignid=1455363063&utm_adgroupid=65083631748&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=1t1&utm_creative=278443377086&utm_targetid=aud-390929969673:dsa-498578051924&utm_loc_interest_ms=&utm_loc_physical_ms=9040331&gclid=CjwKCAjw-4_mBRBuEiwA5xnFIErgo0CmIBG7V3KlWfbC0KVEN6O-NintJH1Mv61puXEMg3mpPDv8vxoCBqEQAvD_BwE)
<br /> [4] [Using TFRecords and tf.Example](https://www.tensorflow.org/alpha/tutorials/load_data/tf_records)

