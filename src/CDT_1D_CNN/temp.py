import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

import os
os.chdir("../..")
from src import *

_df = pd.read_csv("Data/wig20_d.csv")
# df = pd.read_csv("Data/mwig40_d.csv")
# df = pd.read_csv("Data/swig80_d.csv")

_df.drop(range(1000), inplace=True)
_df.reset_index(inplace=True, drop=True)

data = _df.drop(columns=["Data"]).values.transpose()
_x_train, _y_train, _x_test, _y_test, _x_validation, _y_validation = data_prepare_to_model(data, batch_size=3**7)


model = tf.keras.Sequential()

model.add(tf.keras.Input(shape=_x_train.shape[1:]))
for i in range(7):
    model.add(Cdt1dLayer())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding='same'))

model.add(tf.keras.layers.Dense(12))
model.add(tf.keras.layers.Dense(7))
model.add(tf.keras.layers.Dense(3, activation="softmax"))
model.add(tf.keras.layers.Dense(2))

model.build(input_shape=_x_train.shape[1:])
model.summary()
# model = tf.keras.Sequential()
#
# for i in range(7):
#     model.add(Cdt1dLayer())
#     model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding='same'))
#
# model.add(tf.keras.layers.Dense(12))
# model.add(tf.keras.layers.Dense(7))
# model.add(tf.keras.layers.Dense(3, activation="softmax"))
# model.add(tf.keras.layers.Dense(2))

# model.build(input_shape=_x_train.shape)
# model.summary()
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#               loss=tf.keras.losses.BinaryCrossentropy())
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='mean_squared_error')

print(_x_train.shape, _y_train.shape)
# b_size = 32
# model.fit(_x_train[:b_size, :, :, :], _y_train[:b_size, :, :, :])
model.fit(_x_train, _y_train, epochs=100)
model.evaluate(_x_test, _y_test)








