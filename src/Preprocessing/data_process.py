import numpy as np
from typing import Tuple, Union
from sklearn.preprocessing import StandardScaler


class DataProcess:
    """Class for processing stock market time series data"""
    def __init__(
            self,
            data_input: np.ndarray,
            batch_size: int = 60,
            test_ratio: float = 0.15,
            validation_ratio: float = 0.1,
            axis_to_split: int = 1,
            channel_len: int = 2,
            feature_to_predict_num: int = 3,
            threshold_rise: float = 0.02,
            threshold_fall: float = -0.02,
            scale_data: bool = True,
            scale_exclude_rows: list = None,
            balance_training_dataset: bool = True,
            drop_zero_value_rows: bool = True,
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
        self.data_target_row = None

        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio
        self.axis_to_split = axis_to_split
        self.channel_len = channel_len
        self.feature_to_predict_num = feature_to_predict_num
        self.threshold_rise = threshold_rise
        self.threshold_fall = threshold_fall
        self.scale_data = scale_data
        self.scale_exclude_rows = scale_exclude_rows if scale_exclude_rows else []
        self.balance_training_dataset = balance_training_dataset
        self.drop_zero_value_rows = drop_zero_value_rows

    def run(self):
        """
        function with main pipeline for data preprocessing. Can be controlled with class's constructor arguments
        :return: None
        """
        if self.drop_zero_value_rows:
            self.data_input = self._drop_zero_values_rows(self.data_input, self.feature_to_predict_num)
        _data = self._target_variable_add(
            data=self.data_input.copy(),
            feature_to_predict_num=self.feature_to_predict_num,
            threshold_rise=self.threshold_rise,
            threshold_fall=self.threshold_fall
        )
        _train_set, _validation_set, _test_set = self._split_train_test_validation(
            data=_data,
            test_ratio=self.test_ratio,
            validation_ratio=self.validation_ratio,
            axis_to_split=self.axis_to_split
        )

        if self.scale_data:
            _train_set, (_validation_set, _test_set) = self._scale_data(
                fit_dataset=_train_set,
                transform_datasets=(_validation_set, _test_set),
                exclude_rows=self.scale_exclude_rows
            )

        _train_set, _validation_set, _test_set = [
            self._create_channels(_set, self.channel_len) for _set in (_train_set, _validation_set, _test_set)
        ]

        (self._X_train, self._y_train), (self._X_validation, self._y_validation), (self._X_test, self._y_test) = [
            self._batch_windows_create(_set, self.data_target_row, self.batch_size) for _set in (
                _train_set, _validation_set, _test_set
            )
        ]

        if self.balance_training_dataset:
            (self._X_train, self._y_train), (self._X_validation, self._y_validation) = self.balance_dataset_with_batches(
                self._X_train, self._y_train, self._X_validation, self._y_validation
            )

    def get_data(
            self,
            mode: str = "all"
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        function returns preprocessed data stored in class's instance
        :param mode: parameter to control what datasets are being returned
        :return: tuple with desired datasets
        """
        assert mode in ["all", "train", "test", "validation"]

        if mode == "all":
            return self._X_train, self._y_train, self._X_validation, self._y_validation, self._X_test, self._y_test
        if mode == "train":
            return self._X_train, self._y_train
        if mode == "test":
            return self._X_test, self._y_test
        if mode == "validation":
            return self._X_validation, self._y_validation

    @staticmethod
    def _create_channels(
            data: np.ndarray,
            channel_len: int
    ) -> np.ndarray:
        """
        function creates array with 'channel_len' identical channels of 'data' array.
        :param data: numpy array data
        :param channel_len: length of channel to duplicate
        :return: array of shape (*data.shape, channel_len), where all channels contain identical array ('data')
        """
        _result = np.broadcast_to(
            data.reshape([*data.shape, 1]),
            [*data.shape, channel_len]
        )
        return _result

    @staticmethod
    def _split_train_test_validation(
            data: np.ndarray,
            test_ratio: float = 0.1,
            validation_ratio: float = 0.1,
            axis_to_split: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        function splits data into train, validation and test sets with respect to given ratios
        :param data: data array
        :param test_ratio: float from 0 to 1 to indicate what percentage of data will be used for testing
        :param validation_ratio: float from 0 to 1 to indicate what percentage of data will be used for validating
        :param axis_to_split: axis to split data for train, test and validation sets by
        :return: tuple with data split to 3 sets, each with shape (features, (instances*{train|validation|test}ratio)//1)
        """
        _data_len = data.shape[axis_to_split]

        _train_set, _validation_set, _test_set = np.split(
            data,
            [int(_data_len * (1 - validation_ratio - test_ratio)), int(_data_len * (1 - test_ratio))],
            axis=axis_to_split
        )
        return _train_set, _validation_set, _test_set

    @staticmethod
    def _batch_windows_create(
            data_array: np.ndarray,
            data_target_row: int,
            batch_size: int = 60
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        function divides 'data_array' to batches windows of size 'batch_size' by its width and extracts target class
            variable from each batch, which is the last value in each batches' row indicated with 'self.data_target_row'
        :param data_array: array of shape (height, width, channels)
        :param data_target_row: number of row from 'data_array' with prediction values
        :param batch_size: size of batches
        :return: arrays of shapes (batches = width-batch_size+1, height-1, batch_size, channels) and (batches, 3).
            First array contains predictors split to batches of given size and second contains target predictions
                for each batch
        """
        _x = np.array([data_array[:, i:i+batch_size, :].astype("float32") for i in range(data_array.shape[1]-batch_size)])

        _y_class_ind = _x[:, data_target_row, -1, 0]
        _x = np.delete(_x, data_target_row, axis=1)

        _y = np.zeros((_x.shape[0], 3), dtype="int32")
        _y[:, 0][_y_class_ind == -1] = 1
        _y[:, 1][_y_class_ind == 0] = 1
        _y[:, 2][_y_class_ind == 1] = 1
        return _x, _y

    @staticmethod
    def _scale_data(
            fit_dataset: Union[Tuple[np.ndarray, ...], np.ndarray],
            transform_datasets: Union[Tuple[np.ndarray, ...], np.ndarray],
            exclude_rows: list
    ) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
        """
        function for scaling data with sklearn StandardScaler class
        :param fit_dataset: datasets to perform fitting on (and also predictions later)
        :param transform_datasets: datasets to perform only transformations on
        :param exclude_rows: list with row numbers to exclude from scaling
        :return: 2 tuples with input datasets after scaling transformation.
            First tuple contains datasets from 'fit_datasets' and second from 'transform_datasets'
        """
        if type(transform_datasets) != tuple:
            transform_datasets = (transform_datasets,)

        indices_rows = list(set(range(fit_dataset.shape[0])))

        if exclude_rows:
            [indices_rows.remove(i) for i in set(exclude_rows)]

        _fit_dataset = np.transpose(fit_dataset[indices_rows, :])
        scaler = StandardScaler()
        scaler.fit(_fit_dataset)

        for _dataset in [fit_dataset] + list(transform_datasets):
            _dataset[indices_rows, :] = np.transpose(
                scaler.transform(
                    np.transpose(_dataset[indices_rows, :]),
                    copy=True
                )
            )

        return fit_dataset, transform_datasets
    
    @staticmethod
    def _drop_zero_values_rows(
        data: np.ndarray,
        feature_to_predict_num: int,
    ) -> np.ndarray:
        good_rows_mask = ~(data[feature_to_predict_num, :] == 0)
        return data[:, good_rows_mask]
        

    def _target_variable_add(
        self,
        data: np.ndarray,
        feature_to_predict_num: int,
        threshold_rise: float,
        threshold_fall: float
    ) -> np.ndarray:
        """
        function adds row with prediction targets of rises/falls by set threshold of target feature. Number of row
            with prediction targets is saved as instances parameter in 'self.data_target_row'
        :param data: numpy array with data of shape (features, instances)
        :param feature_to_predict_num: number of feature to use for creating prediction targets from first dimension
        :param threshold_rise: percentage rise threshold
        :param threshold_fall: percentage fall threshold
        :return: numpy array with concatenated data and prediction targets of shape (features+1, instances-1)
        """
        _target_row = data[feature_to_predict_num:feature_to_predict_num+1, :]
        _target_row_pct_change = (_target_row[:, 1:] - _target_row[:, :-1]) / _target_row[:, :-1]

        _mask_still = (threshold_fall < _target_row_pct_change) & (_target_row_pct_change < threshold_rise)
        _mask_fall = (_target_row_pct_change <= threshold_fall)
        _mask_rise = (threshold_rise <= _target_row_pct_change)

        _y_row = np.zeros_like(_target_row_pct_change, dtype=int)
        _y_row[_mask_still] = 0
        _y_row[_mask_fall] = -1
        _y_row[_mask_rise] = 1

        result = np.concatenate((data[:, :-1], _y_row), axis=0)
        self.data_target_row = result.shape[0] - 1
        self.scale_exclude_rows += [self.data_target_row]
        return result

    @staticmethod
    def balance_dataset_with_batches(
            x_set_balance: np.ndarray,
            y_set_balance: np.ndarray,
            x_set_fill: np.ndarray,
            y_set_fill: np.ndarray
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        method trims given 'set_balance', so it will contain prediction classes of equal size.
            Cut observations will be added to 'set_fill' dataset
        :param x_set_balance: predictors from set that needs balancing
        :param y_set_balance: predictions from set that needs balancing
        :param x_set_fill: predictors from set that will be filled with trimmed observations
        :param y_set_fill: predictions from set that will be filled with trimmed observations
        :return: 2 tuples with predictors and predictions of datasets,
            one with balanced prediction classes and the other filled with trimmed observations
        """
        shortest_class_len = y_set_balance.sum(axis=0).min()
        class_indexes = np.argmax(y_set_balance, axis=1)

        class_0_len, class_1_len, class_2_len = y_set_balance.sum(axis=0)
        class_0_mask, class_1_mask, class_2_mask = np.zeros(class_0_len, dtype=bool), np.zeros(class_1_len, dtype=bool), np.zeros(class_2_len, dtype=bool)

        class_0_mask[:shortest_class_len] = 1
        class_1_mask[:shortest_class_len] = 1
        class_2_mask[:shortest_class_len] = 1

        class_0_mask, class_1_mask, class_2_mask = [
            np.random.permutation(mask) for mask in (class_0_mask, class_1_mask, class_2_mask)
        ]

        x_class_0_keep, y_class_0_keep, x_class_1_keep, y_class_1_keep, x_class_2_keep, y_class_2_keep = (
            x_set_balance[class_indexes == 0][class_0_mask, ...],
            y_set_balance[class_indexes == 0][class_0_mask, ...],
            x_set_balance[class_indexes == 1][class_1_mask, ...],
            y_set_balance[class_indexes == 1][class_1_mask, ...],
            x_set_balance[class_indexes == 2][class_2_mask, ...],
            y_set_balance[class_indexes == 2][class_2_mask, ...]
        )

        x_class_0_move, y_class_0_move, x_class_1_move, y_class_1_move, x_class_2_move, y_class_2_move = (
            x_set_balance[class_indexes == 0][~class_0_mask, ...],
            y_set_balance[class_indexes == 0][~class_0_mask, ...],
            x_set_balance[class_indexes == 1][~class_1_mask, ...],
            y_set_balance[class_indexes == 1][~class_1_mask, ...],
            x_set_balance[class_indexes == 2][~class_2_mask, ...],
            y_set_balance[class_indexes == 2][~class_2_mask, ...]
        )

        x_keep, y_keep, x_move, y_move = (
            np.concatenate((x_class_0_keep, x_class_1_keep, x_class_2_keep), axis=0),
            np.concatenate((y_class_0_keep, y_class_1_keep, y_class_2_keep), axis=0),
            np.concatenate((x_set_fill, x_class_0_move, x_class_1_move, x_class_2_move), axis=0),
            np.concatenate((y_set_fill, y_class_0_move, y_class_1_move, y_class_2_move), axis=0)
        )

        return (x_keep, y_keep), (x_move, y_move)


__all__ = [
    "DataProcess"
]
