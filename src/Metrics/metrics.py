import numpy as np
import os
import tensorflow as tf

from keras import backend as K
from tensorflow import keras
from typing import Tuple, Union

from metrics_utils import *


beta_1, beta_2, beta_3 = 0.5, 0.125, 0.125


class StorePredictionsCallback(keras.callbacks.Callback):
    def __init__(
        self,
        validation_data: Tuple[np.ndarray, np.ndarray],
        name: str = "epoch_end"
    ) -> None:
        super(StorePredictionsCallback, self).__init__()
        self.validation_data = validation_data
        self.name = name
        
        self.y_predictions = []
        self.y_true = []
        

    def on_epoch_end(self, epoch, logs=None):
        # Get predictions for the current epoch
        y_pred = self.model.predict(self.validation_data[0])

        self.y_predictions.append(y_pred)
        self.y_true.append(self.validation_data[1])
    
    def save(self, base_path: str) -> None:
        np.save(os.path.join(base_path, self.name + "_val_pred.npy"), np.array(self.y_predictions))
        np.save(os.path.join(base_path, self.name + "_val_true.npy"), np.array(self.y_true))

    def __init__(self, engine: str) -> None:
        """
        :param engine: engine for methods to be used. Possible are in FuncEngine.func_engine_dict.keys()
        :type engine: str
        """
        self.argmax = self.__class__.func_engine_dict["argmax"][engine]
        self.compare = self.__class__.func_engine_dict["compare"][engine]
        self.sum = self.__class__.func_engine_dict["sum"][engine]
        self.mul = self.__class__.func_engine_dict["mul"][engine]
        self.logical_and = self.__class__.func_engine_dict["logical_and"][engine]


@confusion_matrix_element_predef(0, 0)
def confusion_matrix_pred_0_true_0(_y_true: Union[np.ndarray, tf.Tensor], _y_pred: Union[np.ndarray, tf.Tensor]):
    """Calculates element in confusion matrix for predicted class 0 and true class 0"""
    pass


@confusion_matrix_element_predef(1, 0)
def confusion_matrix_pred_0_true_1(_y_true: Union[np.ndarray, tf.Tensor], _y_pred: Union[np.ndarray, tf.Tensor]):
    """Calculates element in confusion matrix for predicted class 0 and true class 1"""
    pass


@confusion_matrix_element_predef(2, 0)
def confusion_matrix_pred_0_true_2(_y_true: Union[np.ndarray, tf.Tensor], _y_pred: Union[np.ndarray, tf.Tensor]):
    """Calculates element in confusion matrix for predicted class 0 and true class 2"""
    pass


@confusion_matrix_element_predef(0, 1)
def confusion_matrix_pred_1_true_0(_y_true: Union[np.ndarray, tf.Tensor], _y_pred: Union[np.ndarray, tf.Tensor]):
    """Calculates element in confusion matrix for predicted class 1 and true class 0"""
    pass


@confusion_matrix_element_predef(1, 1)
def confusion_matrix_pred_1_true_1(_y_true: Union[np.ndarray, tf.Tensor], _y_pred: Union[np.ndarray, tf.Tensor]):
    """Calculates element in confusion matrix for predicted class 1 and true class 1"""
    pass


@confusion_matrix_element_predef(2, 1)
def confusion_matrix_pred_1_true_2(_y_true: Union[np.ndarray, tf.Tensor], _y_pred: Union[np.ndarray, tf.Tensor]):
    """Calculates element in confusion matrix for predicted class 1 and true class 2"""
    pass


@confusion_matrix_element_predef(0, 2)
def confusion_matrix_pred_2_true_0(_y_true: Union[np.ndarray, tf.Tensor], _y_pred: Union[np.ndarray, tf.Tensor]):
    """Calculates element in confusion matrix for predicted class 2 and true class 0"""
    pass


@confusion_matrix_element_predef(1, 2)
def confusion_matrix_pred_2_true_1(_y_true: Union[np.ndarray, tf.Tensor], _y_pred: Union[np.ndarray, tf.Tensor]):
    """Calculates element in confusion matrix for predicted class 2 and true class 1"""
    pass


@confusion_matrix_element_predef(2, 2)
def confusion_matrix_pred_2_true_2(_y_true: Union[np.ndarray, tf.Tensor], _y_pred: Union[np.ndarray, tf.Tensor]):
    """Calculates element in confusion matrix for predicted class 2 and true class 2"""
    pass


@confusion_matrix_predef
def confusion_matrix(_y_true, _y_pred, engine):
    func_list = [
        confusion_matrix_pred_0_true_0(engine=engine), confusion_matrix_pred_1_true_0(engine=engine), confusion_matrix_pred_2_true_0(engine=engine),
        confusion_matrix_pred_0_true_1(engine=engine), confusion_matrix_pred_1_true_1(engine=engine), confusion_matrix_pred_2_true_1(engine=engine),
        confusion_matrix_pred_0_true_2(engine=engine), confusion_matrix_pred_1_true_2(engine=engine), confusion_matrix_pred_2_true_2(engine=engine),
    ]
    return func_list


def recall_m(y_true, y_pred):
    """
    :param y_true: _description_
    :type y_true: _type_
    :param y_pred: _description_
    :type y_pred: _type_
    :return: _description_
    :rtype: _type_
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    """

    :param y_true: _description_
    :type y_true: _type_
    :param y_pred: _description_
    :type y_pred: _type_
    :return: _description_
    :rtype: _type_
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    """

    :param y_true: _description_
    :type y_true: _type_
    :param y_pred: _description_
    :type y_pred: _type_
    :return: _description_
    :rtype: _type_
    """
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


@f_1_weighted_predef(beta_1, beta_2, beta_3)
def f1_weighted(
    _y_pred_error_type_1: float,
    _y_pred_error_type_2: int,
    _y_pred_error_type_3: int,
    _y_pred_correct,
    beta_1: float,
    beta_2: float,
):
    """
    Function calculates weighted F-score metric value. Metric elements are calculated inside used decorator.

    :param y_true: Tensor of shape (n_batch, n_class=3) with one-hot indicating true label.
    :param y_pred: Tensor of shape (n_batch, n_class=3) with predicted probabilities for labels.
    :return: Calculated value of weighted F-score metric
    """
    f_score_weighted = ((1 + beta_1**2 + beta_2**2) * _y_pred_correct) \
        / (
            (1 + beta_1**2 + beta_2**2) * _y_pred_correct \
            + _y_pred_error_type_1 \
            + _y_pred_error_type_2 * beta_1**2 \
            + _y_pred_error_type_3 * beta_2**2 \
        )
    return f_score_weighted


__all__ = [
    "recall_m",
    "precision_m",
    "f1_m",
    "f1_weighted",
    "confusion_matrix",
    "confusion_matrix_pred_0_true_0", "confusion_matrix_pred_0_true_1", "confusion_matrix_pred_0_true_2",
    "confusion_matrix_pred_1_true_0", "confusion_matrix_pred_1_true_1", "confusion_matrix_pred_1_true_2",
    "confusion_matrix_pred_2_true_0", "confusion_matrix_pred_2_true_1", "confusion_matrix_pred_2_true_2",
    "StorePredictionsCallback",
]

if __name__ == "__main__":
    pass

