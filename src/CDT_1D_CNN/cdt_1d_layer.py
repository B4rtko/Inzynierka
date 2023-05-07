# Zasotosowanie konwolucji 1d, wraz z przekształceniem danych 2D w 1D po dodaniu kolumny 0, a potem flattowanie

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
import tensorflow.python.keras as krs
from tensorflow.python.keras.layers import LSTM, Dropout, Dense, Layer, Conv1D
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from typing import Union


class Cdt1dLayer(tf.keras.layers.Layer):
    """So far probably working initial layer for CDT-1D CNN without trainable weights""" #  todo update after changes in class

    # def _restore_from_tensors(self, restored_tensors):
    #     pass
    #
    # def _serialize_to_tensors(self):
    #     pass

    def __init__(self):
        super(Cdt1dLayer, self).__init__()
        #  todo write it properly
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

    def build(self, input_shape):
        self.kernel_1 = self.add_weight(
            shape=(3, 1, 1),
            initializer="random_normal",
            trainable=True,
        )
        self.kernel_2 = self.add_weight(
            shape=(3, 1, 1),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs, **kwargs):
        """
        converts input array to flattened array and perform 1D convolution
        :param inputs: array with data
        :return: 2-channeled array after 1D convolution
        """
        debug = False

        inputs_flat = inputs_flat_with_pad(inputs)
        inputs_flat_1 = inputs_flat[:, :, :1]
        inputs_flat_2 = inputs_flat[:, :, -1:]

        res_1 = tf.nn.convolution(input=inputs_flat_1, filters=np.ones((3, 1, 1)), padding="SAME")
        # res_1 = tf.nn.convolution(input=inputs_flat_1, filters=self.kernel_1, padding="SAME")
        res_2 = tf.nn.convolution(input=inputs_flat_2, filters=self.kernel_2, padding="SAME")
        res = tf.stack((res_1, res_2), axis=-1)

        #res = tf.nn.convolution(input=inputs_flat, filters=self.kernel, padding="SAME")
        # res = tf.nn.conv1d(input=inputs_flat, filters=self.kernel, padding="SAME", stride=1)

        if debug:
            print("Shape input", inputs.shape)
            print("Shape input flat", inputs_flat.shape)
            print("Shape result flat", res.shape)

        # result = tf.reshape(res, [res.shape[0], inputs.shape[1], -1, res.shape[2]])[:, :, :-1, :]
        result = tf.reshape(res, [-1, inputs.shape[1], inputs.shape[2]+1, res.shape[2]])[:, :, :-1, :]

        return result


def inputs_flat_with_pad(inputs: np.array):
    """
    Add column of zeros at the end of 2D spatial array and flatten it
    :param inputs: 4D array of data with shape: [batches, y, x, channels]
    :return: 3D array with flattened spatial array with shape: [batches, y*(x+1), channels]
    """
    dim_batches, dim_y, dim_x, dim_channels = inputs.get_shape()

    paddings = [[0, 0], [0, 0], [0, 1], [0, 0]]
    result = tf.pad(inputs, paddings, "CONSTANT")

    result = tf.squeeze(
        # tf.reshape(result, [dim_batches, 1, -1, dim_channels]),
        tf.reshape(result, [-1, 1, result.shape[1]*result.shape[2], dim_channels]),
        axis=1)
    return result


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
    create 3D array with filters for tensorflow convolution where filters values are in 1st dimension and particular filters are stored in 3rd dimension
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


if __name__ == "__main__":
    pass

