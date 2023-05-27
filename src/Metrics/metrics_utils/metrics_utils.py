import functools
import numpy as np
import tensorflow as tf

from typing import Callable, Union
        

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
                _y_pred_arg = f.argmax(y_pred, axis = -1)
                _y_true_arg = f.argmax(y_true, axis = -1)
                
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


__all__ = [
    "FuncEngine",
    "f_1_weighted_predef",
    "confusion_matrix_element_predef",
    "confusion_matrix_predef",
]