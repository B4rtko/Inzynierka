from keras import backend as K
import tensorflow as tf


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
]

if __name__ == "__main__":
    pass

