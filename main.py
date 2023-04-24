import itertools
import multiprocessing
import os
import yaml

from src.Model_training import TrainCDT_1D
from typing import List, Dict


def run(
    _config_file_path: str,
    _model_class=TrainCDT_1D,
    _pool_workers: int = None
):
    """
    Function executes training procedures in multiprocess using model
        of given class. Training is performed for all cases written in
        given config file.

    :param _config_file_path: _description_
    :type _config_file_path: str
    :param _model_class: _description_, defaults to TrainCDT_1D
    :type _model_class: _type_, optional
    :param _pool_workers: _description_, defaults to None
    :type _pool_workers: int, optional
    """
    _pool_workers = multiprocessing.cpu_count() if _pool_workers is None\
        else _pool_workers
    config_cases = config_unpack_cases(_config_file_path)
    
    model_list = [
        _model_class(
            **config,
            dir_path_suffix=str(i)
        ) for (config, i) in enumerate(config_cases)
    ]
    for (model, config) in zip(model_list, config_cases):
        with open(os.path.join(model.model_save_dir, "config.yaml"), 'w') as f:
            yaml.dump(
                config,
                f,
                default_flow_style=False
            )

    with multiprocessing.Pool(_pool_workers) as p:
        p.map(_run_task, model_list)


def config_unpack_cases(
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
    config_cases = [{k: v for (k, v) in zip(config_keys, case)} for case in itertools.product(*config_keys)]
    return config_cases


def _run_task(_model):
    """
    Function to perform '_model.run()' method as a subprocess of multiprocessing.

    :param _model: machine learning model with 'run' method
    """
    _model.run()


if __name__ == '__main__':
    config_file_path = os.path.join("Configs", "Crude_Oil_5.yaml")
    model_class = TrainCDT_1D
    pool_workers = None

    run(
        _config_file_path=config_file_path,
        _model_class=model_class,
        _pool_workers=pool_workers
    )

