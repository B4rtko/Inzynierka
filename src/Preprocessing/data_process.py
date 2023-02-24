import numpy as np
from typing import Tuple


def data_prepare_to_model_2d(
        data_array: np.ndarray,
        batch_size: int = 60,
        test_ratio: float = 0.15,
        validation_ratio: float = 0.1,
        axis_to_split: int = 1,
        channel_len: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    pipeline function to duplicate data for channels, split it into train, test and validation sets and split into batches
    :param data_array: 2D numpy array with data of size (height, width)
    :param batch_size: int for size of batches
    :param test_ratio: float from 0 to 1 to indicate what percentage of data will be used for testing
    :param validation_ratio: float from 0 to 1 to indicate what percentage of data will be used for validating
    :param axis_to_split: axis to split data for train, test and validation sets by
    :param channel_len: size of desired channels
    :return: arrays with x and y data for train, test and validation after:
        - duplicating data for channels
        - splitting data into train, test and validation sets
        - split data sets into window batches
    """
    data_array = _create_channels(data_array, channel_len)
    _train_set, _test_set, _validation_set = _split_train_test_validation(data_array, test_ratio, validation_ratio, axis_to_split)

    _x_train, _y_train = _batches_windows_create_2d(_train_set, batch_size)
    _x_test, _y_test = _batches_windows_create_2d(_test_set, batch_size)
    _x_validation, _y_validation = _batches_windows_create_2d(_validation_set, batch_size)

    return _x_train, _y_train, _x_test, _y_test, _x_validation, _y_validation


def _create_channels(_data: np.ndarray, channel_len: int):
    """
    function creates array with 'channel len' identical channels of '_data' array.
    :param _data: data
    :param channel_len: length of channel to duplicate
    :return: Array of shape (*_data.shape, channel_len), where all channels contain identical array ('_data')
    """
    _result = np.broadcast_to(
        _data.reshape([*_data.shape, 1]),
        [*_data.shape, channel_len]
    )
    return _result


def _split_train_test_validation(
        _data_array: np.ndarray,
        _test_ratio: float = 0.1,
        _validation_ratio: float = 0.1,
        _axis_to_split: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    function splits data into train, test and validation sets with respect to given ratios
    :param _data_array: data array
    :param _test_ratio: float from 0 to 1 to indicate what percentage of data will be used for testing
    :param _validation_ratio: float from 0 to 1 to indicate what percentage of data will be used for validating
    :param _axis_to_split: axis to split data for train, test and validation sets by
    :return: data split to 3 sets
    """
    _data_len = _data_array.shape[_axis_to_split]
    _train_set, _test_set, _validation_set = np.split(
        _data_array,
        [int(_data_len * (1 - _test_ratio - _validation_ratio)), int(_data_len * (1 - _validation_ratio))],
        axis=_axis_to_split
    )
    return _train_set, _test_set, _validation_set


def _batches_windows_create_2d(
        data_array: np.array,
        batch_size: int = 60
):
    """
    function divides 'data_array' to batches windows of size 'batch_size' by its width
    :param data_array: array of shape (height, width, channels)
    :param batch_size: size of batches
    :return: arrays of shapes (batches = width-batch_size+1, height, batch_size, channels) and
        (batches = width-batch_size+1, height, 1, channels)
    """
    _x = np.array([data_array[:, i:i+batch_size, :].astype("float32") for i in range(data_array.shape[1]-batch_size)])
    _y = np.array([data_array[:, i+batch_size:i+batch_size+1, :].astype("float32") for i in range(data_array.shape[1]-batch_size)])
    return _x, _y


def data_prepare_to_model(  # 1D version, not in tests, probably trash
        data_array: np.ndarray,
        batch_size: int = 60,
        test_ratio: float = 0.15,
        validation_ratio: float = 0.1,
        axis_to_split: int = 1,
        channel_len: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """

    :param data_array: 2D numpy array with data
    :param batch_size:
    :param test_ratio:
    :param validation_ratio:
    :param axis_to_split:
    :param channel_len:
    :return:
    """
    data_array = _create_channels(data_array, channel_len)

    _data_len = data_array.shape[axis_to_split]
    _train_set, _test_set, _validation_set = np.split(
        data_array,
        [int(_data_len*(1-test_ratio-validation_ratio)), int(_data_len*(1-validation_ratio))],
        axis=axis_to_split
    )

    _x_train, _y_train = _batches_create(_train_set, batch_size)
    _x_test, _y_test = _batches_create(_test_set, batch_size)
    _x_validation, _y_validation = _batches_create(_validation_set, batch_size)

    return _x_train, _y_train, _x_test, _y_test, _x_validation, _y_validation


def _batches_create(  # 1D version, not in tests, probably trash
        data_array: np.array,
        batch_size: int = 60
):
    """

    :param data_array:
    :param batch_size:
    :return:
    """
    _x = np.array([data_array[:, i:i+batch_size].astype("float32") for i in range(data_array.shape[1]-batch_size-1)])  # probably bad range
    _y = np.array([data_array[:, i+batch_size:i+batch_size+1].astype("float32") for i in range(data_array.shape[1]-batch_size-1)])  # probably bad range
    return _x, _y


__all__ = [
    "data_prepare_to_model",
    "data_prepare_to_model_2d"
]
