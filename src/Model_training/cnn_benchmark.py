import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import tensorflow as tf
import yaml

from datetime import datetime
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from typing import List, Tuple, Union

from src.Metrics import f1_m
from src.Preprocessing import DataProcess

data_filename_dict = {
    "wig_20": ["Data", "wig20_d.csv"],
    "crude_oil_5": ["Data", "2_Extracted", "Crude_Oil_5.csv"],
}

metrics_dict = {
    "f1_m": f1_m,
}


class TrainCNN:
    """Class implements model training pipeline for model with Cross Data Type 1-D convolution layers."""
    def __init__(
        self,
        convolution_layers_count: int = 5,
        dense_layers_units: List[int] = (1000, 500),  # dense layers' units except of last softmax layer
        epoch_count: int = 500,
        data_filename: Union[Tuple[str], List[str], str] = ("Data", "2_Extracted", "Tesla_5.csv"),
        threshold_fall: float = -0.001,
        threshold_rise: float = 0.001,
        learning_rate: float = 1e-4,
        test_ratio: float = 0.07,
        validation_ratio: float = 0.05,
        feature_to_predict_num: int = 3,
        balance_training_dataset: bool = True,
        batch_size: int = None,
        metrics = (f1_m,),
        early_stopping_params = None,
        dir_path_suffix: str = "",
    ):
        self.convolution_layers_count = convolution_layers_count
        self.dense_layers_units = dense_layers_units
        self.epoch_count = epoch_count
        self.data_filename = os.path.join(*data_filename) if type(data_filename) in (list, tuple)\
            else os.path.join(*data_filename_dict[data_filename])
        self.threshold_fall = threshold_fall
        self.threshold_rise = threshold_rise
        self.learning_rate = learning_rate
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio
        self.feature_to_predict_num = feature_to_predict_num
        self.balance_training_dataset = balance_training_dataset
        self.batch_size = 100 if batch_size is None\
            else batch_size
        self.metrics = [
            metrics_dict[m] if type(m)==str else m for m in metrics
        ]
        self.early_stopping_params = early_stopping_params

        self.model_save_dir = os.path.join(
            "Models",
            "CNN",
            os.path.basename(self.data_filename)[:os.path.basename(self.data_filename).find(".")],
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + dir_path_suffix
        )
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_save_dir, "Figures"), exist_ok=True)

        self.model_callbacks = []

        self.data = None
        self.data_pipeline = None
        self._x_train, self._y_train, self._x_validation, self._y_validation, self._x_test, self._y_test = [None for _ in range(6)]
        self._y_test_ind, self._y_validation_ind = None, None
        self.model, self.model_history = None, None

    def run(self):
        """
        Method for executing main pipeline of training process, which contain:
        - data loading,
        - data processing,
        - model building, training and evaluating,
        - saving model with its configuration and scores.
        """
        self._data_actions()
        self._model_actions()

    def _data_actions(self):
        """
        Method for executing methods for data handling.
        """
        self.__data_load()
        self.__data_preprocess()

    def _model_actions(self):
        """
        Method for executing methods for ML model handling.
        """
        self.__model_prepare()
        self.__model_fit()
        self.__model_evaluate_validation()
        self.__model_evaluate_test()
        self.__model_history_to_csv()
        self.__model_training_history_visualise()

    def __data_load(self):
        """
        Method:
        - loads dataframe from 'self.data_filename',
        - drops 'Data' column if exists,
        - get numpy array from values,
        - transposes array,
        - saves result to 'self.data' attribute.
        """
        _df = pd.read_csv(self.data_filename)
        self.data = _df.drop(columns=["Data"], errors="ignore").values.transpose()

    def __data_preprocess(self):
        """
        Method:
        - creates 'DataProcess' instance and saves it as attribute,
        - processes 'self.data' with provided parameters,
        - saves result datasets to attributes.
        """
        self.data_pipeline = DataProcess(
            self.data,
            test_ratio = self.test_ratio,
            validation_ratio = self.validation_ratio,
            batch_size = self.batch_size,
            threshold_fall = self.threshold_fall,
            threshold_rise = self.threshold_rise,
            feature_to_predict_num = self.feature_to_predict_num,
            balance_training_dataset = self.balance_training_dataset
        )
        self.data_pipeline.run()
        self._x_train, self._y_train, self._x_validation, self._y_validation, self._x_test, self._y_test = self.data_pipeline.get_data()
        self._y_test_ind, self._y_validation_ind = np.argmax(self._y_test, axis=1), np.argmax(self._y_validation, axis=1)

    def __model_prepare(self):
        """
        Method for building and compiling model with provided parameters.
        """
        self.model = tf.keras.Sequential()

        self.model.add(tf.keras.Input(shape=self._x_train.shape[1:]))
        for _ in range(self.convolution_layers_count):
            self.model.add(
                tf.keras.layers.Conv2D(2, (3, 3), 1, padding="same")
            )
            self.model.add(
                tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')
            )

        self.model.add(tf.keras.layers.Flatten())
        for units in self.dense_layers_units:
            self.model.add(tf.keras.layers.Dense(units, activation="relu"))
        self.model.add(tf.keras.layers.Dense(3, activation="softmax"))

        self.model.build(input_shape=self._x_train.shape[1:])

        # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        #               loss=f1_m,
        #               metrics=f1_m
        #               )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=self.metrics,
            # run_eagerly=True,
        )

    def __model_fit(self):
        """
        Method for training the model with training data and saving fitted result.
        Model training history is being saved to 'self.model_history' attribute
        """

        if self.early_stopping_params:
            _es = tf.keras.callbacks.EarlyStopping(
                monitor = self.early_stopping_params["monitor"],
                mode = self.early_stopping_params["mode"],
                min_delta = self.early_stopping_params["min_delta"],
                patience = self.early_stopping_params["patience"],
                start_from_epoch = self.early_stopping_params["start_from_epoch"],
            )
            self.model_callbacks.append(_es)
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

        self.model_history = self.model.fit(
            self._x_train,
            self._y_train,
            epochs=self.epoch_count,
            callbacks=self.model_callbacks,
            validation_data=(self._x_validation, self._y_validation),
        )

        self.model.save(os.path.join(self.model_save_dir, "Model"))

    def __model_evaluate_validation(self):
        """
        Method performs evaluation of model performance on validation dataset and saves the scores.
        Current performance measures:
        - confusion matrix
        """
        # self.model.evaluate(self._x_validation, self._y_validation)
        pred_validation = self.model.predict(self._x_validation)
        pred_validation_ind = np.argmax(pred_validation, axis=1)

        confusion_matrix(self._y_validation_ind, pred_validation_ind)

    def __model_evaluate_test(self):
        """
        Method performs evaluation of model performance on test dataset and saves the scores.
        Current performace measures:
        - confusion matrix
        """
        # self.model.evaluate(self._x_test, self._y_test)
        pred_test = self.model.predict(self._x_test)
        pred_test_ind = np.argmax(pred_test, axis=1)

        confusion_matrix(self._y_test_ind, pred_test_ind)
        
    def __model_history_to_csv(self):
        pd.DataFrame(self.model_history.history).to_csv(
            os.path.join(self.model_save_dir, "Metrics_and_losses.csv"),
            index=False,
            errors="ignore",
        )

    def __model_training_history_visualise(self):
        """
        Method visualises model's loss function and metrics during training process.
        """
        for _metric in self.model_history.history.keys():
            _fig, ax = plt.subplots(figsize=(10,10))
            sns.lineplot(x=np.arange(len(self.model_history.history[_metric]))+1, y=self.model_history.history[_metric], ax=ax, label=_metric)
            _fig.savefig(os.path.join(self.model_save_dir, "Figures", _metric + ".png"))


__all__ = [
    "TrainCNN",
]


if __name__ == "__main__":
    pass
