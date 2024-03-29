import itertools
import multiprocessing
import os
import yaml

from src.Model_training import TrainCDT_1D, TrainCNN
from typing import List, Dict


def run(
    _config_mode: str,
    _data_name: str,
    _model_class=TrainCDT_1D,
    _pool_workers: int = None
):
    """
    Function executes training procedures in multiprocess using model
        of given class. Training is performed for all cases written in
        given config file.

    :param _config_cases: _description_
    :param _model_class: _description_, defaults to TrainCDT_1D
    :type _model_class: _type_, optional
    :param _pool_workers: _description_, defaults to None
    :type _pool_workers: int, optional
    """
    _config_cases = _get_configs_by_mode(
        _config_mode,
        _data_name,
    )

    _pool_workers = multiprocessing.cpu_count() if _pool_workers is None\
        else _pool_workers


    model_list = [
        _model_class(
            **config,
            dir_path_suffix=str(i)
        ) for (i, config) in enumerate(_config_cases)
    ]
    for (model, config) in zip(model_list, _config_cases):
        with open(os.path.join(model.model_save_dir, "config.yaml"), 'w') as f:
            yaml.dump(
                config,
                f,
                default_flow_style=False
            )
    
    with multiprocessing.Pool(_pool_workers) as p:
        p.map(_run_task, model_list)


def _config_unpack_cases(
    _config_file_path: str
) -> List[Dict]:
    """
    Function reads configuration from given yaml file and splits it for config cases.

    :param _config_file_path: _description_
    :type _config_file_path: str
    """
    with open(_config_file_path) as f:
        config_params = yaml.safe_load(f)
    
    config_keys = list(config_params.keys())
    _config_cases_generator = itertools.product(*[config_params[k] for k in config_keys])
    config_cases = [dict(zip(config_keys, case)) for case in _config_cases_generator]
    return config_cases


def _run_task(_model):
    """
    Function to perform '_model.run()' method as a subprocess of multiprocessing.

    :param _model: machine learning model with 'run' method
    """
    _model.run()


def _config_get_from_directory(
    _directory_path
) -> List[Dict]:
    _config_yaml_list = []
    _config_cases_list = []
    
    for root, dirs, files in os.walk(_directory_path):
        for f in files:
            if f.endswith(".yaml") or f.endswith(".yml"):
                _config_yaml_list.append(os.path.join(root, f))
    if not _config_yaml_list:
        raise Exception("No config cases found")
    
    for _config_yaml_path in _config_yaml_list:
        with open(_config_yaml_path) as f:
            _config_cases_list.append(yaml.safe_load(f))
    return _config_cases_list
    
    

def _get_configs_by_mode(
    _mode: str,
    _data_name: str,
) -> List[Dict]:
    _args = _get_config_args_by_mode(_mode, _data_name)

    if _mode == "from_yaml":
        return _config_unpack_cases(*_args)
    elif _mode == "from_uncompleted":
        return _config_get_from_directory(*_args)
    else:
        raise Exception(f"Unknown mode: '{_mode}'")


def _get_config_args_by_mode(
    _mode: str,
    _data_name: str,
) -> List:
    if _mode == "from_yaml":
        return [
            os.path.join("Configs", _data_name + ".yaml"),
        ]
    elif _mode == "from_uncompleted":
        return [
            os.path.join("Models", "CDT_1D", _data_name, "Uncompleted"),
        ]
    else:
        raise Exception(f"Unknown mode: '{_mode}'")



if __name__ == '__main__':
    data_name = \
    "Amazon_5"
    "Apple_5"
    "Crude_Oil_5"
    "DJI_5"
    "Tesla_5"

    config_mode = \
    "from_yaml"
    "from_uncompleted"

    model_class = \
    TrainCNN    
    TrainCDT_1D

    pool_workers = \
    1
    4
    2
    None
    

    run(
        _config_mode=config_mode,
        _data_name=data_name,
        _model_class=model_class,
        _pool_workers=pool_workers
    )

