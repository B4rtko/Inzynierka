import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras import backend as K

import os
from src import *
from Metrics import f1_m


def _run(
    convolution_layers_count = 7,
    epoch_count = 10000
):
    ### Load data

    # data_filename_list = ["Data", "wig20_d.csv"]
    # data_filename_list = ["Data", "2_Extracted", "Crude_Oil_5.csv"]
    data_filename_list = ["Data", "2_Extracted", "Tesla_5.csv"]

    _df = pd.read_csv(os.path.join(*data_filename_list), index_col=0)
    # _df = pd.read_csv("Data/wig20_d.csv")
    # _df = pd.read_csv("Data/mwig40_d.csv")
    # _df = pd.read_csv("Data/swig80_d.csv")

    # _df.drop(range(1000), inplace=True)
    _df.reset_index(inplace=True, drop=True)
    _df.head()
    data = _df.drop(columns=["Data"], errors="ignore").values.transpose()

    ### Process and split data

    # thr_fall, thr_rise = (-0.005270574305918364, 0.004968199728502841)
    thr_fall, thr_rise = (-0.001, 0.001)

    data_pipeline = DataProcess(
        data, test_ratio=0.07, validation_ratio=0.05, batch_size=3**(convolution_layers_count-2),
        threshold_fall=thr_fall, threshold_rise=thr_rise, feature_to_predict_num=3,
        balance_training_dataset=True
    )
    data_pipeline.run()
    _x_train, _y_train, _x_validation, _y_validation, _x_test, _y_test = data_pipeline.get_data()
    _y_test_ind, _y_validation_ind = np.argmax(_y_test, axis=1), np.argmax(_y_validation, axis=1)



    ### Model
    #### Building

    model = tf.keras.Sequential()

    model.add(tf.keras.Input(shape=_x_train.shape[1:]))
    for i in range(convolution_layers_count):
        model.add(tf.keras.layers.Conv2D(2, (1, 3), 1, padding="same"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding='same'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1000))
    model.add(tf.keras.layers.Dense(500))
    model.add(tf.keras.layers.Dense(3, activation="softmax"))

    model.build(input_shape=_x_train.shape[1:])

    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    #               loss=f1_m,
    #               metrics=f1_m
    #               )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=f1_m
                )

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                        mode='min',
                                        min_delta=1e-3,
                                        patience=500,
                                        start_from_epoch=300,
                                        )
    # es = tf.keras.callbacks.EarlyStopping(monitor='val_f1_m',
    #                                       mode='max',
    #                                       min_delta=1e-3,
    #                                       patience=200,
    #                                       start_from_epoch=200,
    #                                       )
    # mc = tf.keras.callbacks.ModelCheckpoint('Models/cdt_2/best_model.h5',
    #                                         monitor='val_loss',
    #                                         mode='min'
    #                                         )
    # mc = tf.keras.callbacks.ModelCheckpoint('Models/cdt_2/best_model.h5',
    #                                         monitor='val_f1_m',
    #                                         mode='max'
    #                                         )

    history = model.fit(_x_train, _y_train,
                        epochs=epoch_count,
                    # callbacks=[es, mc],
                    callbacks=[es],
                    validation_data=(_x_validation, _y_validation),
                        )

    model.save(os.path.join("Models", "CDT_1D", data_filename_list[-1][:-4]))

    ### Evaluating
    #### Validation

    model.evaluate(_x_validation, _y_validation)
    pred_validation = model.predict(_x_validation)
    pred_validation_ind = np.argmax(pred_validation, axis=1)

    confusion_matrix(_y_validation_ind, pred_validation_ind)
    #### Test

    model.evaluate(_x_test, _y_test)
    pred_test = model.predict(_x_test)
    pred_test_ind = np.argmax(pred_test, axis=1)

    confusion_matrix(_y_test_ind, pred_test_ind)
    history.history.keys()
    fig, ax = plt.subplots(2, 1, figsize=(15,10))
    sns.lineplot(x=np.arange(len(history.history["loss"]))+1, y=history.history["loss"], ax=ax[0], label="Cross entropy loss")
    sns.lineplot(x=np.arange(len(history.history["f1_m"]))+1, y=history.history["f1_m"], ax=ax[1], label="F1 score")
    fig.savefig()
