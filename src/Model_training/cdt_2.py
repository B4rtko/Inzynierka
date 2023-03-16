# so far unused

import os
import pandas as pd
import tensorflow as tf
from src import *


def run(
        data_csv_path: str,
        save_name: str = None
):
    """
    pipeline function to run model training
    :param data_csv_path: path with data for training
    :param save_name: if given, the model will be saved in 'Models' directory
    :return: None
    """
    _df = pd.read_csv(os.path.join("..", "..", "Data", data_csv_path))

    _df.drop(range(1000), inplace=True)
    _df.reset_index(inplace=True, drop=True)

    data = _df.drop(columns=["Data"]).values.transpose()

    _x_train, _y_train, _x_test, _y_test, _x_validation, _y_validation = data_prepare_to_model_2d(data, test_ratio=0.2, validation_ratio=0.2, batch_size=32)
    print("train", _x_train.shape)
    print("test", _x_test.shape)
    print("validation", _x_validation.shape)

    print("train", _y_train.shape)
    print("test", _y_test.shape)
    print("validation", _y_validation.shape)

    model = tf.keras.Sequential()

    model.add(tf.keras.Input(shape=_x_train.shape[1:]))
    for i in range(7):
        model.add(tf.keras.layers.Conv2D(2, (1, 3), 1, padding="same"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding='same'))

    model.add(tf.keras.layers.Dense(12))
    model.add(tf.keras.layers.Dense(7))
    model.add(tf.keras.layers.Dense(3, activation="softmax"))
    model.add(tf.keras.layers.Dense(2))

    model.build(input_shape=_x_train.shape[1:])
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='mean_squared_error')
    model.fit(_x_train, _y_train, epochs=10)
    model.evaluate(_x_test, _y_test)
    if not save_name:
        model.save(os.path.join("..", "..", "Models", save_name))


if __name__ == "__main__":
    run("wig20_d.csv", save_name=None)
