# Image Analysis with Long Short-TermMemory Recurrent Neural Networks
This artist for studying the Ph.D disertation of Dr Wonmin Byeon [Image Analysis with Long Short-TermMemory Recurrent Neural Networks](https://pdfs.semanticscholar.org/ccdd/6874aa8924152d0ad4a74a37542def74eff0.pdf?_ga=2.23143610.1241662214.1570246157-609034475.1570246157),
and also for understant some applications of [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) in computer vision fields.
In this artist I also tried to made some example with [LSTM]() with [tensorflow]() framework.

# Basic knowledge

## Long short-term Memory
Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. It can not only process single data points (such as images), but also entire sequences of data (such as speech or video). For example, LSTM is applicable to tasks such as unsegmented, connected handwriting recognition or speech recognition. Bloomberg Business Week wrote: "These powers make LSTM arguably the most commercial AI achievement, used for everything from predicting diseases to composing music.

<b> LSTM </b>:

1. Cell
2. input gate
3. ouput gate
4. forget gate

The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell. 

LSTM networks are well-suited to classifying, processing and making predictions based on time series data, since there can be lags of unknown duration between important events in a time series. LSTMs were developed to deal with the exploding and vanishing gradient problems that can be encountered when training traditional RNNs. Relative insensitivity to gap length is an advantage of LSTM over RNNs, hidden Markov models and other sequence learning methods in numerous applications.

 

# References

[LSTM - Wiki](https://en.wikipedia.org/wiki/Long_short-term_memory)

[Image Analysis with Long Short-TermMemory Recurrent Neural Networks](https://pdfs.semanticscholar.org/ccdd/6874aa8924152d0ad4a74a37542def74eff0.pdf?_ga=2.23143610.1241662214.1570246157-609034475.1570246157)

[Recurrent Neural Networks](http://slazebni.cs.illinois.edu/fall18/lec15_rnn.pdf)


