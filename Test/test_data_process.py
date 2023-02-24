import pytest
import numpy as np

from src import data_prepare_to_model
from src.Preprocessing.data_process import _batches_create, _create_channels, _split_train_test_validation

arrays_template = {
    "input_1": np.arange(20).reshape((4, 5)),
    "channeled_1": np.array([
        [[0,  0], [1,  1], [2,  2], [3,  3], [4,  4]],
        [[5,  5], [6,  6], [7,  7], [8,  8], [9,  9]],
        [[10, 10], [11, 11], [12, 12], [13, 13], [14, 14]],
        [[15, 15], [16, 16], [17, 17], [18, 18], [19, 19]]
    ]),
    "splitted_1": {
        "train": np.array([[[0,  0], [1,  1], [2,  2]],
                           [[5,  5], [6,  6], [7,  7]],
                           [[10, 10], [11, 11], [12, 12]],
                           [[15, 15], [16, 16], [17, 17]]]
                          ),
        "test": np.array([[[3, 3]],
                          [[8, 8]],
                          [[13, 13]],
                          [[18, 18]]]
                         ),
        "validation": np.array([[[4, 4]],
                                [[9, 9]],
                                [[14, 14]],
                                [[19, 19]]]
                               )
    },
    "splitted_2": {
        "train": np.array([[[0, 0], [1, 1]],
                           [[5, 5], [6, 6]],
                           [[10, 10], [11, 11]],
                           [[15, 15], [16, 16]]]
                          ),
        "test": np.array([[[2, 2]],
                          [[7, 7]],
                          [[12, 12]],
                          [[17, 17]]]
                         ),
        "validation": np.array([[[3, 3], [4, 4]],
                                [[8, 8], [9, 9]],
                                [[13, 13], [14, 14]],
                                [[18, 18], [19, 19]]]
                               )
    }

}

test_params_create_channels = {
    "input_1": (arrays_template["input_1"], 2),
    "output_1": arrays_template["channeled_1"]
}

test_params_split_train_test_validation = {
    "input_1": {"_data_array": arrays_template["channeled_1"],
                "_test_ratio": 0.2,
                "_validation_ratio": 0.2
                },
    "output_1": arrays_template["splitted_1"],
    "input_2": {"_data_array": arrays_template["channeled_1"],
                "_test_ratio": 0.25,
                "_validation_ratio": 0.25
                },
    "output_2": arrays_template["splitted_2"]
}

test_params = test_params_create_channels


@pytest.mark.parametrize("inputs, expected", [
    (test_params["input_1"], test_params["output_1"])
])
def test__create_channels(inputs, expected):
    assert np.array_equal(_create_channels(*inputs), expected)


test_params = test_params_split_train_test_validation


@pytest.mark.parametrize("inputs, expected", [
    (test_params["input_1"], test_params["output_1"]),
    (test_params["input_2"], test_params["output_2"])
])
def test__split_train_test_validation(inputs, expected):
    results = _split_train_test_validation(**inputs)
    assert all(
        (np.array_equal(results[0], expected["train"]),
         np.array_equal(results[1], expected["test"]),
         np.array_equal(results[2], expected["validation"])
         )
    )




