import pytest
import numpy as np
import tensorflow as tf

from src import Cdt1dLayer, inputs_flat_with_pad, backup_inputs_flat_with_pad, filter_combine


arrays_template = {
    "input_1": tf.constant(np.arange(20).reshape([1, 4, -1, 1]).astype("float32")),
    "input_2": tf.constant(np.arange(40).reshape([1, 4, -1, 1]).astype("float32")),
    "input_3": tf.constant(np.arange(120).reshape([1, 4, -1, 1]).astype("float32")),
    # "input_2": tf.constant(np.moveaxis(np.arange(40).reshape([1, 2, 4, 5]), 1, 3).astype("float32")),  # probably some channels
    # "input_3": tf.constant(np.moveaxis(np.arange(120).reshape([3, 2, 4, 5]), 1, 3).astype("float32")),  # probably some channels
    "flatten_with_pad_1": tf.constant(np.array([
        [0, 1, 2, 3, 4, 0],
        [5, 6, 7, 8, 9, 0],
        [10, 11, 12, 13, 14, 0],
        [15, 16, 17, 18, 19, 0]
    ]).flatten().reshape([1, -1, 1])),
    "flatten_with_pad_2": tf.constant(np.moveaxis(
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
    )),
    "flatten_with_pad_3": tf.constant(np.moveaxis(
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
    )),
    "convolved_1": tf.constant(np.array([[1, 3, 6, 9, 7],
                                         [11, 18, 21, 24, 17],
                                         [21, 33, 36, 39, 27],
                                         [31, 48, 51, 54, 37]]
                                        ).reshape([1, 4, -1, 1]).astype("float32")),
    "convolved_2": tf.constant(np.array([[1, 3, 6, 9, 12, 15, 18, 21, 24, 17],
                                         [21, 33, 36, 39, 42, 45, 48, 51, 54, 37],
                                         [41, 63, 66, 69, 72, 75, 78, 81, 84, 57],
                                         [61, 93, 96, 99, 102, 105, 108, 111, 114, 77]]
                                        ).reshape([1, 4, -1, 1]).astype("float32")),
    "convolved_3": tf.constant(np.array([
        [1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 57],
        [61, 93,  96,  99, 102, 105, 108, 111, 114, 117, 120, 123, 126, 129, 132, 135, 138, 141, 144, 147, 150, 153, 156, 159, 162, 165, 168, 171, 174, 117],
        [121, 183, 186, 189, 192, 195, 198, 201, 204, 207, 210, 213, 216, 219, 222, 225, 228, 231, 234, 237, 240, 243, 246, 249, 252, 255, 258, 261, 264, 177],
        [181, 273, 276, 279, 282, 285, 288, 291, 294, 297, 300, 303, 306, 309, 312, 315, 318, 321, 324, 327, 330, 333, 336, 339, 342, 345, 348, 351, 354, 237]]
    ).reshape([1, 4, -1, 1]).astype("float32"))
}

test_params_inputs_flat_with_pad = {
    "input_1": arrays_template["input_1"],
    "output_1": arrays_template["flatten_with_pad_1"],
    "input_2": arrays_template["input_2"],
    "output_2": arrays_template["flatten_with_pad_2"],
    "input_3": arrays_template["input_3"],
    "output_3": arrays_template["flatten_with_pad_3"]
}

test_params_convolution_2d = {
    "input_1": (arrays_template["input_1"], tf.constant(np.ones((1, 3, 1, 1)).astype("float32"))),
    "output_1": arrays_template["convolved_1"],
    "input_2": (arrays_template["input_2"], tf.constant(np.ones((1, 3, 1, 1)).astype("float32"))),
    "output_2": arrays_template["convolved_2"],
    "input_3": (arrays_template["input_3"], tf.constant(np.ones((1, 3, 1, 1)).astype("float32"))),
    "output_3": arrays_template["convolved_3"]
}

param_dict = test_params_inputs_flat_with_pad


@pytest.mark.parametrize("inputs, expected", [
    (param_dict["input_1"], param_dict["output_1"]),
    (param_dict["input_2"], param_dict["output_2"]),
    (param_dict["input_3"], param_dict["output_3"])
])
def test_inputs_flat_with_pad(inputs, expected):
    assert np.array_equal(inputs_flat_with_pad(inputs), expected)


param_dict = test_params_convolution_2d


@pytest.mark.parametrize("inputs, expected", [
    (param_dict["input_1"], param_dict["output_1"]),
    (param_dict["input_2"], param_dict["output_2"]),
    (param_dict["input_3"], param_dict["output_3"])
])
def test_inputs_convolution_2d(inputs, expected):
    assert np.array_equal(
        tf.nn.conv2d(input=inputs[0], filters=inputs[1], padding="SAME", strides=1),  # todo ude cdt_1d class
        expected
    )





