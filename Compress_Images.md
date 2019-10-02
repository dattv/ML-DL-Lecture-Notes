#Compress Images
This artist is devoted to re-note some information about compressing Images by Artificial Neural Network

[Using AI to Super Compress Images](https://hackernoon.com/using-ai-to-super-compress-images-5a948cf09489)

[An End-to-End Compression Framework Based onConvolutional Neural Networks](https://arxiv.org/pdf/1708.00838v1.pdf)

[A  Neural Network based Technique for Data Compression](https://pdfs.semanticscholar.org/3d3a/ef65d58cc2668e95948e567cbd357daff3a8.pdf)

[Full Resolution Image Compression with RecurrentNeural Networks](https://arxiv.org/pdf/1608.05148v1.pdf)

[Lossless Data Compression with Neural Networks](https://bellard.org/nncp/nncp.pdf)

[DeepZip: Lossless Data Compression usingRecurrent Neural Networks](https://arxiv.org/pdf/1811.08162.pdf)

[Neuralnetwork and Image compression - Stanford](https://cs.stanford.edu/people/eroberts/courses/soco/projects/neural-networks/Applications/imagecompression.html)

[image-compression-benchmarking - github](https://github.com/arassadin/image-compression-benchmarking)

[https://github.com/Justin-Tan/generative-compression - github](https://github.com/Justin-Tan/generative-compression)

[Generative Adversarial Networks for Extreme Learned Image Compression](https://arxiv.org/pdf/1804.02958.pdf)

## What is image compression ?
Image compression is the process of converting an image so that it occupies less space. Simply storing the images would take up a lot of space, so there are codecs, such as JPEG and PNG that aim to reduce the size of the original image.

## Lossy vs. Lossless compression
There are two types of image compression :Lossless and Lossy. As their names suggest, in Lossless compression, it is possible to get back all the data of the original image, while in Lossy, some of the data is lost during the convsersion.
(<b>JPG</b> is a lossy algorithm, while <b>PNG</b> is a lossless algorithm. <b>Losless is good, but it ends up taking a lot of space on disk</b>).

There are better ways to compress images without losing much information, but they are quite slow, and many use iterative approaches, which means they cannot be run in parallel over multiple CPU cores , or GPUs. This renders them quite impractical in everyday usage.

## Convolutional Neural Network
If anything needs to be computed and it can be approximated , throw a neural network at it. The authors used a fairly standard Convolutional Neural Network to improve image compression. Their method not only performs at par with the ‘better ways’ <b>(if not even better), it can also leverage parallel computing, resulting in a dramatic speed increase</b>.

The reasoning behind it is that convolution neural networks(CNN) are very good at extracting spatial information from images, which are then represented in a more compact form (e.g. only the ‘important’ bits of an image are stored). The authors wanted to leverage this capability of CNNs to be able to better represent images


## The Architecture
The authors proposed a dual network. The first network , which will take the image and generate a compact representation(ComCNN). The output of this network will then be processed by a standard codec (e.g. JPEG). After going through the codec, the image will be passed to a 2nd network, which will ‘fix’ the image from the codec, trying to get back the original image. The authors called it Reconstructive CNN (RecCNN). Both networks are iteratively trained, similar to a GAN.

The output from the codec is upscaled , then passed to RecCNN. The RecCNN will try to output an image that looks as similar to original image as possible.

