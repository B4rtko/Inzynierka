# Podejście z wykorzystaniem konwolucji 2d, w której jeden z wymiarów jest równy 1, a drugi 3

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


class Cdt1dLayer2(tf.keras.layers.Layer):
    """Keras layer for performing Cross-Data-Type 1D Convolution using flat 2D filter"""

    # def _restore_from_tensors(self, restored_tensors):
    #     pass
    #
    # def _serialize_to_tensors(self):
    #     pass

    def __init__(self):
        super(Cdt1dLayer2, self).__init__()
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
        perform CDT 1D Convolution using flatten 2D convolution
        :param inputs: array with data
        :return: 2-channeled array after 1D convolution
        """
        filter = tf.constant(np.ones((1, 3, 1, 1)).astype("float32"))
        result = tf.nn.conv2d(input=inputs, filters=filter, padding="SAME", strides=1)

        return result


__all__ = [
    "Cdt1dLayer2"
]


if __name__ == "__main__":
    pass

