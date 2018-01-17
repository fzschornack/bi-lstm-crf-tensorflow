# bi-lstm-crf

The notebook `bi-lstm-crf-tensorflow.ipynb` contains an example of a Bidirectional LSTM + CRF (Conditional Random Fields) model in Tensorflow.

I tried to keep the problem and implementation as simple as possible so anyone can understand and change the model to meet their own problem and data.

And to make it more realistic, the inputs have variable sequence lengths.

## Sequence Classification Problem

We will define a simple sequence classification problem to explore bidirectional LSTMs + CRF.

The problem is defined as a sequence of random values between 0 and 1. 

A binary label (0 or 1) is associated with each input. Initially, the output values are all 0. Once the cumulative sum of the input values in the sequence exceeds a threshold, then the output value flips from 0 to 1.

A threshold of 1/4 the sequence length is used.

For example, below is a sequence of 10 input timesteps (X):

```python
0.63144003 0.29414551 0.91587952 0.95189228 0.32195638 0.60742236 0.83895793 0.18023048 0.84762691 0.29165514
```

In this case the threshold is `2.5` and the corresponding classification output (y) would be:

```python
0 0 0 1 1 1 1 1 1 1
```

## Notes on bidirectional-LSTM and CRF Tensorflow structures

Both `bidirectional_dynamic_rnn` and `crf_log_likelihood` use the optional `sequence_length` parameter.

This parameter holds the real `sequence lengths` of the inputs (without the padding) and, when running the model, TensorFlow will return zero vectors for states and outputs after these sequence lengths. 

Therefore, weights will not get trained on the padding information.

\*Obs.: The padding is necessary to use batches in Tensorflow, in order to speed up the computations.

## References

1. Tutorial this code was based on: [Bi-LSTM + CRF with character embeddings for NER and POS](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html)
2. Bidirectional LSTM in keras (contains the description of the problem used in this code): [How to Develop a Bidirectional LSTM For Sequence Classification in Python with Keras](https://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/)
3. [Example of Bidirectional LSTM implementation in Tensorflow](https://stackoverflow.com/questions/39808336/tensorflow-bidirectional-dynamic-rnn-none-values-error)
4. [Example of CRF implementation in Tensorflow](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/crf)
5. Explains in details the LSTM "num_units" parameter: [Understanding LSTM in Tensorflow](https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/)
6. Explains how variable sequence lengths work in Tensorflow: [Variable Sequence Lengths in TensorFlow](https://danijar.com/variable-sequence-lengths-in-tensorflow/)