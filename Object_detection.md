# Object detection

Object detection had an explosion about finding the position of the object in addition to labeling the object is called object localization. Typically, the position of the object is defined by rectangular coordinates. Finding multiple objects in the image with rectangular coordinates is called detection

## Methods


### Anchor based methods

<b> 1 </b> [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)

#### Improved - crowded objects
<b> 1 </b> [Anchor Pruning for Object Detection - 2020](https://arxiv.org/pdf/2104.00432.pdf)

<b> 2 </b> [RMOPP: Robust Multi-Objective Post-Processing for Effective Object Detection - 2021](https://arxiv.org/pdf/2102.04582.pdf)

<b> 3 </b> [LLA: Loss-aware Label Assignment for Dense Pedestrian Detection - 2021](https://arxiv.org/pdf/2101.04307.pdf)

### Anchor Free methods

## Dataset

<b>1</b>  [ImageNet dataset -  There are 200 objects for detection problems with 470,000 images, with an average of 1.1 objects per image]()

<b>2</b> [PASCAL VOC challenge - There are 20 classes in the dataset. The dataset has 11,530 images for training and validations with 27,450 annotations for regions of interest]()


1. ```Person```: Person
2. ```Animal```: Bird, cat, cow, dog, horse, sheep
3. ```Vehicle```: Airplane, bicycle, boat, bus, car, motorbike, train
4. ```Indoor```: Bottle, chair, dining table, potted plant, sofa, tv/monitor
5. There is an average of 2.4 objects per image.

<b>3</b> [COCO object detection - dataset has 200,000 images with more than 500,000 object annotations in 80 categories]()

1. The average number of objects is 7.2 per image.
