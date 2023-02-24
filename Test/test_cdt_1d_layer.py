import pytest
import numpy as np

from src import Cdt1dLayer, inputs_flat_with_pad, backup_inputs_flat_with_pad, filter_combine


parameters_test_inputs_flat_with_pad = {
    "input_data_1": np.arange(20).reshape([1, 4, 5, 1]),
    "input_data_2": np.moveaxis(np.arange(40).reshape([1, 2, 4, 5]), 1, 3),
    "input_data_3": np.moveaxis(np.arange(120).reshape([3, 2, 4, 5]), 1, 3),
    "correct_output_1": np.array([
        [0, 1, 2, 3, 4, 0],
        [5, 6, 7, 8, 9, 0],
        [10, 11, 12, 13, 14, 0],
        [15, 16, 17, 18, 19, 0]
    ]).flatten().reshape([1, -1, 1]),
    "correct_output_2": np.moveaxis(
        np.array([
            [[0, 1, 2, 3, 4, 0],
             [5, 6, 7, 8, 9, 0],
             [10, 11, 12, 13, 14, 0],
             [15, 16, 17, 18, 19, 0]],
            [[20, 21, 22, 23, 24, 0],
             [25, 26, 27, 28, 29, 0],
             [30, 31, 32, 33, 34, 0],
             [35, 36, 37, 38, 39, 0]]
        ]).flatten().reshape([1, 2, -1]),
        1, 2
    ),
    "correct_output_3": np.moveaxis(
        np.array([
            [[0, 1, 2, 3, 4, 0],
             [5, 6, 7, 8, 9, 0],
             [10, 11, 12, 13, 14, 0],
             [15, 16, 17, 18, 19, 0]],
            [[20, 21, 22, 23, 24, 0],
             [25, 26, 27, 28, 29, 0],
             [30, 31, 32, 33, 34, 0],
             [35, 36, 37, 38, 39, 0]],
            [[40, 41, 42, 43, 44, 0],
             [45, 46, 47, 48, 49, 0],
             [50, 51, 52, 53, 54, 0],
             [55, 56, 57, 58, 59, 0]],
            [[60, 61, 62, 63, 64, 0],
             [65, 66, 67, 68, 69, 0],
             [70, 71, 72, 73, 74, 0],
             [75, 76, 77, 78, 79, 0]],
            [[80, 81, 82, 83, 84, 0],
             [85, 86, 87, 88, 89, 0],
             [90, 91, 92, 93, 94, 0],
             [95, 96, 97, 98, 99, 0]],
            [[100, 101, 102, 103, 104, 0],
             [105, 106, 107, 108, 109, 0],
             [110, 111, 112, 113, 114, 0],
             [115, 116, 117, 118, 119, 0]]
        ]).flatten().reshape([3, 2, -1]),
        1, 2
    )
}

param_dict = parameters_test_inputs_flat_with_pad


@pytest.mark.parametrize("inputs, expected", [
    (param_dict["input_data_1"], param_dict["correct_output_1"]),
    (param_dict["input_data_2"], param_dict["correct_output_2"]),
    (param_dict["input_data_3"], param_dict["correct_output_3"])
])
def test_inputs_flat_with_pad(inputs, expected):
    assert np.array_equal(inputs_flat_with_pad(inputs), expected)





