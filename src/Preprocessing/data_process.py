import numpy as np
from typing import Tuple


class DataProcess:
    """Class for processing stock market time series data"""
    def __init__(
            self,
            data_input: np.ndarray,
            batch_size: int = 60,
            test_ratio: float = 0.15,
            validation_ratio: float = 0.1,
            axis_to_split: int = 1,
            channel_len: int = 2
    ):
        """
        :param data_input: 2D numpy array with data of size (height, width)
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
        self.data_input = data_input
        self._X_train, self._y_train = None, None
        self._X_test, self._y_test = None, None
        self._X_validation, self._y_validation = None, None

        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio
        self.axis_to_split = axis_to_split
        self.channel_len = channel_len

    def run(self):
        _data = self.data_input
        _data = self._create_channels(_data, self.channel_len)
        _train_set, _test_set, _validation_set = self._split_train_test_validation(_data, self.test_ratio, self.validation_ratio, self.axis_to_split)

        self._X_train, self._y_train = self._batch_windows_create(_train_set, self.batch_size)
        self._X_test, self._y_test = self._batch_windows_create(_test_set, self.batch_size)
        self._X_validation, self._y_validation = self._batch_windows_create(_validation_set, self.batch_size)

    def get_data(self, mode: str = "all"):
        assert mode in ["all", "train", "test", "validation"]

        if mode == "all":
            return self._X_train, self._y_train, self._X_test, self._y_test, self._X_validation, self._y_validation
        if mode == "train":
            return self._X_train, self._y_train
        if mode == "test":
            return self._X_test, self._y_test
        if mode == "validation":
            return self._X_validation, self._y_validation

    @staticmethod
    def _create_channels(
            _data: np.ndarray,
            channel_len: int
    ):
        """
        function creates array with 'channel len' identical channels of '_data' array.
        :param _data: numpy array data
        :param channel_len: length of channel to duplicate
        :return: Array of shape (*_data.shape, channel_len), where all channels contain identical array ('_data')
        """
        _result = np.broadcast_to(
            _data.reshape([*_data.shape, 1]),
            [*_data.shape, channel_len]
        )
        return _result

    @staticmethod
    def _split_train_test_validation(
            _data: np.ndarray,
            _test_ratio: float = 0.1,
            _validation_ratio: float = 0.1,
            _axis_to_split: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        function splits data into train, test and validation sets with respect to given ratios
        :param _data: data array
        :param _test_ratio: float from 0 to 1 to indicate what percentage of data will be used for testing
        :param _validation_ratio: float from 0 to 1 to indicate what percentage of data will be used for validating
        :param _axis_to_split: axis to split data for train, test and validation sets by
        :return: data split to 3 sets
        """
        _data_len = _data.shape[_axis_to_split]
        _train_set, _test_set, _validation_set = np.split(
            _data,
            [int(_data_len * (1 - _test_ratio - _validation_ratio)), int(_data_len * (1 - _validation_ratio))],
            axis=_axis_to_split
        )
        return _train_set, _test_set, _validation_set

    def _batch_windows_create(
            self,
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


__all__ = [
    "DataProcess"
]
