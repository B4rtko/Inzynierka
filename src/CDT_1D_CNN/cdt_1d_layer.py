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
from typing import Union, Tuple


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
        self.kernel = self.add_weight(
            shape=(3, 1, 2),
            initializer="random_normal",
            trainable=True,
        )

    #     #  todo write it properly
    #     self.w1 = self.add_weight(
    #         shape=(input_shape[-1], self.units),
    #         initializer="random_normal",
    #         trainable=True,
    #     )
    #     self.w2 = self.add_weight(
    #         shape=(self.units,),
    #         initializer="random_normal",
    #         trainable=True
    #     )

    def call(self, inputs, **kwargs):
        """
        converts input array to flattened array and perform 1D convolution
        :param inputs: array with data
        :return: 2-channeled array after 1D convolution
        """
        debug = False

        inputs_flat = inputs_flat_with_pad(inputs)

        # filt_1 = [1, 1, 1]
        # filt_2 = [0.5, 0.5, 0.5]
        # filt = np.array([filt_1, filt_2]).transpose().reshape([3, 1, 2])

        # filt = np.zeros([3, 1, 2])
        # filt[:, 0, :1] = self.kernel_1
        # filt[:, 0, 1:] = self.kernel_2

        res = tf.nn.convolution(input=inputs_flat, filters=self.kernel, padding="SAME")

        if debug:
            print("Shape input", inputs.shape)
            print("Shape input flat", inputs_flat.shape)
            # print("Shape filter", filt.shape)
            print("Shape result flat", res.shape)

        # result = res.reshape([res.shape[0], inputs.shape[1], -1, res.shape[2]])[:, :, :-1, :]
        result = tf.reshape(res, [res.shape[0], inputs.shape[1], -1, res.shape[2]])[:, :, :-1, :]

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
        tf.reshape(result, [dim_batches, 1, -1, dim_channels]),
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


def _batches_create(data_array: np.array, batch_size: int = 60):
    _x = np.array([data_array[:, i:i+batch_size].astype("float32") for i in range(data_array.shape[1]-batch_size-1)])
    _y = np.array([data_array[:, i+batch_size].astype("float32") for i in range(data_array.shape[1]-batch_size-1)])
    return _x, _y


def _create_channels(_data: np.ndarray, channel_len: int):
    _result = np.broadcast_to(
        _data.reshape([*_data.shape, 1]),
        [*_data.shape, channel_len]
    )
    return _result


def data_prepare_to_model(
        data_array: np.ndarray,
        batch_size: int = 60,
        test_ratio: float = 0.15,
        validation_ratio: float = 0.1,
        axis_to_split: int = 1,
        channel_len: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """

    :param data_array: 2D numpy array with data
    :param batch_size:
    :param test_ratio:
    :param validation_ratio:
    :param axis_to_split:
    :param channel_len:
    :return:
    """
    data_array = _create_channels(data_array, channel_len)

    _data_len = data_array.shape[axis_to_split]
    _train_set, _test_set, _validation_set = np.split(
        data_array,
        [int(_data_len*(1-test_ratio-validation_ratio)), int(_data_len*(1-validation_ratio))],
        axis=axis_to_split
    )

    _x_train, _y_train = _batches_create(_train_set, batch_size)
    _x_test, _y_test = _batches_create(_test_set, batch_size)
    _x_validation, _y_validation = _batches_create(_validation_set, batch_size)

    return _x_train, _y_train, _x_test, _y_test, _x_validation, _y_validation


__all__ = [
    "Cdt1dLayer",
    "inputs_flat_with_pad",
    "backup_inputs_flat_with_pad",
    "filter_combine",
    "data_prepare_to_model"
]


if __name__ == "__main__":
    pass

