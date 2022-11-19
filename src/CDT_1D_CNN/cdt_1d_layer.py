import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.python.keras as krs
from keras.layers import LSTM, Dropout, Dense, Layer, Conv1D
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from typing import Union


class Cdt1dLayer(Layer):
    """So far probably working initial layer for CDT-1D CNN without trainable weights""" #  todo update after changes in class
    def __init__(self, units=32):
        #  todo write it properly
        super(Cdt1dLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        #  todo write it properly
        self.w1 = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.w2 = self.add_weight(
            shape=(self.units,),
            initializer="random_normal",
            trainable=True
        )
        # self.w = self.add_weight(
        #     shape=(input_shape[-1], self.units),
        #     initializer="random_normal",
        #     trainable=True,
        # )
        # self.b = self.add_weight(
        #     shape=(self.units,),
        #     initializer="random_normal",
        #     trainable=True
        # )

    def call(self, inputs):
        """
        converts input array to flattened array and perform 1D convolution
        :param inputs: array with data
        :return: 2-channeled array after 1D convolution
        """
        inputs_flat = inputs_flat_with_pad(inputs)
        filt = filter_combine([1, 1, 1], [0.5, 0.5, 0.5], in_channels=inputs_flat.shape[2])

        res = tf.nn.convolution(input=inputs_flat, filters=filt, padding="SAME").numpy()
        print("Shape input", inputs.shape)
        print("Shape input flat", inputs_flat.shape)
        print("Shape filter", filt.shape)
        print("Shape result", res.shape)

        # res_1 = res[:, :, :1].reshape((inputs.shape[0],-1,1))[:,:-1,:]
        # res_2 = res[:, :, 1:].reshape((inputs.shape[0],-1,1))[:,:-1,:]

        result = res.reshape((inputs.shape[0], -1, filt.shape[2]))[:,:-1,:]
        return result


def inputs_flat_with_pad(inputs: np.array):
    """
    Add 0 column at the end and flatten to 3D array where 2nd dimension contains flattened values
    :param inputs: 2D array of data
    :return: 3D array with flattened values in 2nd dimension
    """
    inputs = inputs if type(inputs) == np.ndarray else np.array(inputs)
    inputs_3rd_shape = 1 if inputs.ndim == 2 else inputs.shape[2]

    res = np.zeros((inputs.shape[0], inputs.shape[1]+1, inputs_3rd_shape))
    res[:, :-1, :] = inputs.reshape(inputs.shape[0], inputs.shape[1], inputs_3rd_shape)
    res = res.reshape((1, -1, inputs_3rd_shape))#[:,:-1,:]
    return res


def backup_inputs_flat_with_pad(inputs: np.array):
    """
    Add 0 column at the end and flatten to 3D array where 2nd dimension contains flattened values
    :param inputs: 2D array of data
    :return: 3D array with flattened values in 2nd dimension
    """
    res = np.zeros((inputs.shape[0], inputs.shape[1]+1))
    res[:, :-1] = inputs
    res = res.reshape((1, -1, 1))#[:,:-1,:]
    return res


def filter_combine(*weights: Union[list, np.array], in_channels=1):
    """
    create 3D array with filters for tensorflow convulsion where filters values are in 1st dimension and particular filters are stored in 3rd dimension
    :param in_channels: number of channels of input data
    :param weights: lists or correct dimension arrays of weights
    :return: array with filters to convulsion
    """
    out_channels = in_channels*len(weights)
    weights = [np.array(w).reshape((-1, 1, 1)) if type(w) == list else w for w in weights]
    filt = np.zeros((len(weights[0]), in_channels, out_channels))

    for k in range(0, out_channels, len(weights)):
        for i, w in enumerate(weights):
            filt[:, :, k+i: k+i+1] = np.repeat(w, in_channels, axis=1)

    return filt


__all__ = [
    "Cdt1dLayer",
    "inputs_flat_with_pad",
    "backup_inputs_flat_with_pad",
    "filter_combine"
]