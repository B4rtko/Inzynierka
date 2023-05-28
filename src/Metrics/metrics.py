import functools
import numpy as np
import os
import tensorflow as tf

from keras import backend as K
from tensorflow import keras
from typing import Callable, Tuple, Union


beta_1, beta_2, beta_3 = 0.5, 0.125, 0.125


########### Utils ###########
class FuncEngine:
    """Class that ships same method interface for various engines."""
    func_engine_dict = {
        "argmax": {
            "numpy": lambda x: np.argmax(x, axis=-1),
            "tensorflow": lambda x: tf.cast(tf.math.argmax(x, axis=-1), dtype=tf.int32),
        },
        "sum": {
            "numpy": lambda x: np.sum(x),
            "tensorflow": lambda x: tf.math.reduce_sum(x),
        },
        "compare": {
            "numpy": lambda x, y: x==y,
            "tensorflow": lambda x: tf.math.reduce_sum(x),
        },
        "logical_and": {
            "numpy": lambda x, y: np.logical_and(x, y),
            "tensorflow": lambda x, y: tf.logical_and(x, y),
        },
        "mul": {
            "numpy": lambda x, y: x*y,
            "tensorflow": lambda x, y: x*y,
        },
        "array": {
            "numpy": np.array,
            "tensorflow": tf.constant,
        },
        "reshape": {
            "numpy": lambda x, shape_tuple: x.reshape(*shape_tuple),
            "tensorflow": lambda x, shape_tuple: x.reshape(*shape_tuple)  # not sure
        }
    }

    def __init__(self, engine: str) -> None:
        """
        :param engine: engine for methods to be used. Possible are in FuncEngine.func_engine_dict.keys()
        :type engine: str
        """
        self.argmax = self.__class__.func_engine_dict["argmax"][engine]
        self.sum = self.__class__.func_engine_dict["sum"][engine]
        self.compare = self.__class__.func_engine_dict["compare"][engine]
        self.logical_and = self.__class__.func_engine_dict["logical_and"][engine]
        self.mul = self.__class__.func_engine_dict["mul"][engine]
        self.array = self.__class__.func_engine_dict["array"][engine]
        self.reshape = self.__class__.func_engine_dict["reshape"][engine]


def confusion_matrix_element_predef(
    _true_matrix_index: int,
    _pred_matrix_index: int,
    engine: str = "numpy"
) -> Callable:
    """
    Decorator factory for creating function that calculates confusion matrix's element with given indices.
        There is possibility to choose engine for operation methods.

    :param _true_matrix_index: Index for true class in confusion matrix
    :param _pred_matrix_index: Index for prediction class in confusion matrix
    :param engine: Engine for operation methods, defaults to "numpy"
    :return: Decorated function
    """
    def _confusion_matrix_element_predef(func):
        def _set_engine(*args, engine: str = "numpy", **kwargs):
            """
            Function gives possibility to both call wrapped function with args, kwargs and operations' engine
            and to call just with operators' engine to generate wrapped function that will take args and kwargs.

            :param engine: Engine for operation methods, defaults to "numpy"
            :return: Generated function with selected engine or result of called function with selected engine
            """
            if_run_as_engine_setter = (len(args) == 0 and len(kwargs.keys()) == 0)
            f = FuncEngine(engine)

            @functools.wraps(func)
            def wrapped_func(
                y_true: Union[np.ndarray, tf.Tensor],
                y_pred: Union[np.ndarray, tf.Tensor], 
            ) -> int:
                """
                Function wrapper for calculation element of confusion matrix with given indices.

                :param y_true: Array or Tensor with (n_batch, n_class=3) shape containing one-hot on true class.
                :param y_pred: Array or Tensor with (n_batch, n_class=3) shape containing prediction probabilities
                    for belonging to each class
                :return: Confusion matrix value calculated as result of wrapped function
                """
                _y_pred_arg = f.argmax(y_pred)
                _y_true_arg = f.argmax(y_true)

                _y_pred_confusion_mask = f.compare(_y_pred_arg, _pred_matrix_index)
                _y_true_confusion_mask = f.compare(_y_true_arg, _true_matrix_index)

                return f.sum(f.mul(_y_true_confusion_mask, _y_pred_confusion_mask))

            if if_run_as_engine_setter:
                return wrapped_func
            else:
                return wrapped_func(*args, **kwargs)

        return _set_engine
    return _confusion_matrix_element_predef


def confusion_matrix_predef(func: Callable) -> Callable:
    """
    Decorator factory for creating function that calculates confusion matrix.
        There is possibility to choose engine for operation methods.

    :param func: Function to wrap with chosen engine's operations
    :return: Decorated function
    """
    def _set_engine(*args, engine: str = "numpy", **kwargs):
        """
        Function gives possibility to both call wrapped function with args, kwargs and operations' engine
        and to call just with operators' engine to generate wrapped function that will take args and kwargs.

        :param engine: Engine for operation methods, defaults to "numpy"
        :return: Generated function with selected engine or result of called function with selected engine
        """
        if_run_as_engine_setter = (len(args) == 0 and len(kwargs.keys()) == 0)
        f = FuncEngine(engine)

        @functools.wraps(func)
        def wrapped_func(
            y_true: Union[np.ndarray, tf.Tensor],
            y_pred: Union[np.ndarray, tf.Tensor], 
        ) -> int:
            """
            Function wrapper for calculation confusion matrix. Operations are being made with given engine.

            :param y_true: Array or Tensor with (n_batch, n_class=3) shape containing one-hot on true class.
            :param y_pred: Array or Tensor with (n_batch, n_class=3) shape containing prediction probabilities
                for belonging to each class
            :return: Confusion matrix calculated as result of wrapped function
            """
            func_list = func(*args, **kwargs, engine=engine)
            confusion_list = [_func(y_true, y_pred) for _func in func_list]
            return f.reshape(f.array(confusion_list), (3,3))

        if if_run_as_engine_setter:
            return wrapped_func
        else:
            return wrapped_func(*args, **kwargs)
    return _set_engine


def f_1_weighted_predef(
    beta_1: float,
    beta_2: float,
    beta_3: float,
) -> Callable:
    """
    Decorator factory for creating F1 weighted metric function with custom beta parameters and
        possibility to choose engine for operation methods.

    :param beta_1: weighting parameter for 2nd type error, defaults to beta_1
    :param beta_2: weighting parameter for 3nd type error, defaults to beta_2
    :param beta_3: weighting parameter for True Flat, defaults to beta_3
    :return: Function decorator
    """
    def _f_1_weighted_predef(func: Callable) -> Callable:
        """Function returned by decorator factory that will wrap the main function."""
        def _set_engine(*args, engine: str = "numpy", **kwargs):
            """
            Function gives possibility to both call wrapped function with args, kwargs and operations' engine
            and to call just with operators' engine to generate wrapped function that will take args and kwargs.

            :param engine: Engine for operation methods, defaults to "numpy"
            :return: Generated function with selected engine or result of called function with selected engine
            """
            if_run_as_engine_setter = (len(args) == 0 and len(kwargs.keys()) == 0)
            f = FuncEngine(engine)

            @functools.wraps(func)
            def wrapped_func(
                y_true: Union[np.ndarray, tf.Tensor],
                y_pred: Union[np.ndarray, tf.Tensor], 
            ) -> float:
                """
                Function wrapper for calculation elements of F1 weighted metric.

                :param y_true: Array or Tensor with (n_batch, n_class=3) shape containing one-hot on true class.
                :param y_pred: Array or Tensor with (n_batch, n_class=3) shape containing prediction probabilities
                    for belonging to each class
                :return: F1 weighted metric calculated as result of wrapped function with calculated parameter elements
                """
                _y_pred_arg = f.argmax(y_pred)
                _y_true_arg = f.argmax(y_true)
                
                _y_pred_down_mask = f.compare(_y_pred_arg, 0)
                _y_pred_flat_mask = f.compare(_y_pred_arg, 1)
                _y_pred_up_mask = f.compare(_y_pred_arg, 2)
                
                _y_true_down_mask = f.compare(_y_true_arg, 0)
                _y_true_flat_mask = f.compare(_y_true_arg, 1)
                _y_true_up_mask = f.compare(_y_true_arg, 2)
                

                _y_pred_up_true_up = f.logical_and(_y_pred_up_mask, _y_true_up_mask)
                _y_pred_flat_true_flat = f.logical_and(_y_pred_flat_mask, _y_true_flat_mask)
                _y_pred_down_true_down = f.logical_and(_y_pred_down_mask, _y_true_down_mask)

                _y_pred_correct = f.sum(_y_pred_up_true_up) \
                    + f.sum(_y_pred_down_true_down) \
                    + f.sum(_y_pred_flat_true_flat) * beta_3**2
                

                _y_pred_up_true_down = f.logical_and(_y_pred_up_mask, _y_true_down_mask)
                _y_pred_down_true_up = f.logical_and(_y_pred_down_mask, _y_true_up_mask)

                _y_pred_error_type_1 = f.sum(_y_pred_up_true_down) \
                    + f.sum(_y_pred_down_true_up)
                

                _y_pred_up_true_flat = f.logical_and(_y_pred_up_mask, _y_true_flat_mask)
                _y_pred_down_true_flat = f.logical_and(_y_pred_down_mask, _y_true_flat_mask)

                _y_pred_error_type_2 = f.sum(_y_pred_up_true_flat) \
                    + f.sum(_y_pred_down_true_flat)
                

                _y_pred_flat_true_down = f.logical_and(_y_pred_flat_mask, _y_true_down_mask)
                _y_pred_flat_true_up = f.logical_and(_y_pred_flat_mask, _y_true_up_mask)

                _y_pred_error_type_3 = f.sum(_y_pred_flat_true_down) \
                    + f.sum(_y_pred_flat_true_up)

                return func(
                    _y_pred_error_type_1,
                    _y_pred_error_type_2,
                    _y_pred_error_type_3,
                    _y_pred_correct,
                    beta_1, beta_2,
                )
            if if_run_as_engine_setter:
                return wrapped_func
            else:
                return wrapped_func(*args, **kwargs)

        return _set_engine
    return _f_1_weighted_predef


########### Metrics ###########

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
def confusion_matrix(engine):
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


########## Callbacks ##########

class StorePredictionsCallback(keras.callbacks.Callback):
    def __init__(
        self,
        validation_data: Tuple[np.ndarray, np.ndarray],
        name: str = "epoch_end",
        engine: str = "numpy",
    ) -> None:
        super(StorePredictionsCallback, self).__init__()
        self.validation_data = validation_data
        self.name = name
        
        self.y_predictions = []
        self.y_true = []
        

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.validation_data[0])
        y_true = self.validation_data[1]

        self.y_predictions.append(y_pred)
        self.y_true.append(y_true)
    
    def save(self, base_path: str) -> None:
        np.save(os.path.join(base_path, self.name + "_val_pred.npy"), np.array(self.y_predictions))
        np.save(os.path.join(base_path, self.name + "_val_true.npy"), np.array(self.y_true))


class StoreConfusionMatrixCallback(keras.callbacks.Callback):
    def __init__(
        self,
        validation_data: Tuple[np.ndarray, np.ndarray],
        name: str = "epoch_end_confusion_matrix",
        engine: str = "numpy",
    ) -> None:
        super(StoreConfusionMatrixCallback, self).__init__()
        self.validation_data = validation_data
        self.name = name
        
        self.confusion_matrix_func = confusion_matrix(engine=engine)
        self.confusion_matrix_list = []
        

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.validation_data[0])
        y_true = self.validation_data[1]
        
        self.confusion_matrix_list.append(self.confusion_matrix_func(y_true, y_pred))
    
    def save(self, base_path: str) -> None:
        np.save(os.path.join(base_path, self.name + ".npy"), np.array(self.confusion_matrix_list))


class StoreF1WeightedCallback(keras.callbacks.Callback):
    def __init__(
        self,
        validation_data: Tuple[np.ndarray, np.ndarray],
        name: str = "epoch_end_f1_weighted",
        engine: str = "numpy",
    ) -> None:
        super(StoreF1WeightedCallback, self).__init__()
        self.validation_data = validation_data
        self.name = name
        
        self.f1_weighted_func = f1_weighted(engine=engine)
        self.f1_weighted_list = []
        

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.validation_data[0])
        y_true = self.validation_data[1]
        
        self.f1_weighted_list.append(self.f1_weighted_func(y_true, y_pred))
    
    def save(self, base_path: str) -> None:
        np.save(os.path.join(base_path, self.name + ".npy"), np.array(self.f1_weighted_list))


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
    "StoreConfusionMatrixCallback",
    "StoreF1WeightedCallback",
]

if __name__ == "__main__":
    pass

