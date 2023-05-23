import functools
import tensorflow as tf
import numpy as np

import keras
from keras import backend as K


beta_1, beta_2, beta_3 = 0.5, 0.125, 0.125


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


def confusion_matrix_element_predef(_pred_matrix_index, _true_matrix_index):
    def _confusion_matrix_element_predef(func):
        @functools.wraps(func)
        def wrapped_func(y_true, y_pred):
            _y_pred_arg = tf.math.argmax(y_pred, axis = -1)
            _y_true_arg = tf.math.argmax(y_true, axis = -1)

            _y_pred_confusion_mask = _y_pred_arg == _pred_matrix_index
            _y_true_confusion_mask = _y_true_arg == _true_matrix_index

            return func(_y_pred_confusion_mask, _y_true_confusion_mask)
        return wrapped_func
    return _confusion_matrix_element_predef


def _confusion_matrix_value_calculation(_y_pred_mask, _y_true_mask):
    # return tf.math.count_nonzero(tf.logical_and(_y_pred_mask, _y_true_mask))
    # return K.sum(tf.cast(tf.logical_and(_y_pred_mask, _y_true_mask), tf.int64))
    return tf.cast(K.sum(tf.cast(tf.logical_and(_y_pred_mask, _y_true_mask), tf.int64)), tf.int64)


# @confusion_matrix_element_predef(0, 0)
# def confusion_matrix_pred_0_true_0(_y_pred_down_mask, _y_true_down_mask):
#     return _confusion_matrix_value_calculation(_y_pred_down_mask, _y_true_down_mask)

# def confusion_matrix_pred_0_true_0(y_true, y_pred):
#     # return K.sum(K.round(K.clip(K.constant([1., 0., 0.]) * y_pred, 0, 1)))
#     return type(y_true)

def confusion_matrix_pred_0_true_0(y_true, y_pred):
    y_true, y_pred = np.argmax(y_true.numpy(), axis=-1), np.argmax(y_pred.numpy(), axis=-1)
    return np.sum((y_true == 0) & (y_pred == 0))
    

def confusion_matrix_pred_0_true_1(y_true, y_pred):
    return K.sum(y_pred)

# class confusion_matrix_pred_0_true_0(keras.callbacks.Callback):
#     def __init__(self, model, x_test, y_test):
#         self.model = model
#         self.x_test = x_test
#         self.y_test = y_test

#     def on_epoch_end(self, epoch, logs={}):
#         y_pred = self.model.predict(self.x_test, self.y_test)
#         print('confusion_0_0: ', y_pred)



# @confusion_matrix_element_predef(0, 1)
# def confusion_matrix_pred_0_true_1(_y_pred_down_mask, _y_true_flat_mask):
#     return _confusion_matrix_value_calculation(_y_pred_down_mask, _y_true_flat_mask)


@confusion_matrix_element_predef(0, 2)
def confusion_matrix_pred_0_true_2(_y_pred_down_mask, _y_true_up_mask):
    return _confusion_matrix_value_calculation(_y_pred_down_mask, _y_true_up_mask)


@confusion_matrix_element_predef(1, 0)
def confusion_matrix_pred_1_true_0(_y_pred_flat_mask, _y_true_down_mask):
    return _confusion_matrix_value_calculation(_y_pred_flat_mask, _y_true_down_mask)


@confusion_matrix_element_predef(1, 1)
def confusion_matrix_pred_1_true_1(_y_pred_flat_mask, _y_true_flat_mask):
    return _confusion_matrix_value_calculation(_y_pred_flat_mask, _y_true_flat_mask)


@confusion_matrix_element_predef(1, 2)
def confusion_matrix_pred_1_true_2(_y_pred_flat_mask, _y_true_up_mask):
    return _confusion_matrix_value_calculation(_y_pred_flat_mask, _y_true_up_mask)


@confusion_matrix_element_predef(2, 0)
def confusion_matrix_pred_2_true_0(_y_pred_up_mask, _y_true_down_mask):
    return _confusion_matrix_value_calculation(_y_pred_up_mask, _y_true_down_mask)


@confusion_matrix_element_predef(2, 1)
def confusion_matrix_pred_2_true_1(_y_pred_up_mask, _y_true_flat_mask):
    return _confusion_matrix_value_calculation(_y_pred_up_mask, _y_true_flat_mask)


@confusion_matrix_element_predef(2, 2)
def confusion_matrix_pred_2_true_2(_y_pred_up_mask, _y_true_up_mask):
    return _confusion_matrix_value_calculation(_y_pred_up_mask, _y_true_up_mask)


def f1_weighted(y_true, y_pred):
    """
    Function calculates weighted F-score metric value for Keras model training.
    Function can be used with multilabel classification for stock market prediction with 3 possible categories,
        where first indicates price falling, second price staying at the same level and third price rising.

    :param y_true: Tensor of shape (n_batches, 3) with one-hot indicating true label.
    :param y_pred: Tensor of shape (n_batches, 3) with predicted probabilities for labels.
    :return: Calculated value of weighted F-score metric.
    """
    global beta_1, beta_2, beta_3

    _y_pred_arg = tf.math.argmax(y_pred, axis = -1)
    _y_true_arg = tf.math.argmax(y_true, axis = -1)
    
    _y_pred_down_mask = _y_pred_arg == 0
    _y_pred_flat_mask = _y_pred_arg == 1
    _y_pred_up_mask = _y_pred_arg == 2
    
    _y_true_down_mask = _y_true_arg == 0
    _y_true_flat_mask = _y_true_arg == 1
    _y_true_up_mask = _y_true_arg == 2
    
    _y_pred_up_true_up = tf.logical_and(_y_pred_up_mask, _y_true_up_mask)
    _y_pred_flat_true_flat = tf.logical_and(_y_pred_flat_mask, _y_true_flat_mask)
    _y_pred_down_true_down = tf.logical_and(_y_pred_down_mask, _y_true_down_mask)
    _y_pred_correct = tf.reduce_sum(tf.cast(_y_pred_up_true_up, tf.float64)) \
        + tf.reduce_sum(tf.cast(_y_pred_down_true_down, tf.float64)) \
        + tf.reduce_sum(tf.cast(_y_pred_flat_true_flat, tf.float64)) * beta_3**2
    
    _y_pred_up_true_down = tf.logical_and(_y_pred_up_mask, _y_true_down_mask)
    _y_pred_down_true_up = tf.logical_and(_y_pred_down_mask, _y_true_up_mask)
    _y_pred_error_type_1 = tf.reduce_sum(tf.cast(_y_pred_up_true_down, tf.float64)) \
        + tf.reduce_sum(tf.cast(_y_pred_down_true_up, tf.float64))
    
    _y_pred_up_true_flat = tf.logical_and(_y_pred_up_mask, _y_true_flat_mask)
    _y_pred_down_true_flat = tf.logical_and(_y_pred_down_mask, _y_true_flat_mask)
    _y_pred_error_type_2 = tf.reduce_sum(tf.cast(_y_pred_up_true_flat, tf.float64)) \
        + tf.reduce_sum(tf.cast(_y_pred_down_true_flat, tf.float64))
    
    _y_pred_flat_true_down = tf.logical_and(_y_pred_flat_mask, _y_true_down_mask)
    _y_pred_flat_true_up = tf.logical_and(_y_pred_flat_mask, _y_true_up_mask)
    _y_pred_error_type_3 = tf.reduce_sum(tf.cast(_y_pred_flat_true_down, tf.float64)) \
        + tf.reduce_sum(tf.cast(_y_pred_flat_true_up, tf.float64))
    
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
    "confusion_matrix_pred_0_true_0", "confusion_matrix_pred_0_true_1", "confusion_matrix_pred_0_true_2",
    "confusion_matrix_pred_1_true_0", "confusion_matrix_pred_1_true_1", "confusion_matrix_pred_1_true_2",
    "confusion_matrix_pred_2_true_0", "confusion_matrix_pred_2_true_1", "confusion_matrix_pred_2_true_2",
]

if __name__ == "__main__":
    pass

