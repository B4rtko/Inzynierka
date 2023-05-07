import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

from keras.models import load_model
from typing import List, Dict, Tuple, Union
from datetime import datetime

from src.Model_training.cdt_1d import metrics_dict
from src.Model_comparing.modules import *


class ModelComparer:
    def __init__(
        self,
        base_path: str = os.path.join("Models"),
        
    ) -> None:
        self.base_path = base_path

        self.model_paths = self._get_available_paths_with_models(self.base_path)
        self.model_config_metrics_dict = dict()
        
    def populate_model_config_metrics_dict(self):
        self.model_config_metrics_dict = self._get_model_config_metrics_dict(self.model_paths)
    
    def compare_metrics_of_one_model(
        self,
        metric_1: str = "loss",
        metric_2: str = "f1_m",
        config_params_to_title: List = None,
        subplots_kwargs: dict = None,
        seaborn_kwargs: dict = None,
        colormap: Union[str, mpl.colors.ListedColormap] = None,
    ) -> plt.figure:
        comparer = ModelComparerMetricsOneModel(
            base_path=self.base_path,
            model_config_metrics_dict=self.model_config_metrics_dict,
            metric_1=metric_1,
            metric_2=metric_2,
            config_params_to_title=config_params_to_title,
            subplots_kwargs=subplots_kwargs,
            seaborn_kwargs=seaborn_kwargs,
            colormap=colormap
        )
        return comparer.generate_plot()
    
    def compare_metric_between_models(
        self,
        metric: str,
        config_params_to_label: List = None,
        save_path: str = None,
        subplots_kwargs: dict = None,
        seaborn_kwargs: dict = None,
        colormap: Union[str, mpl.colors.ListedColormap] = None,
    ) -> plt.figure:
        comparer = ModelComparerMetricBetweenModels(
            metric=metric,
            base_path=self.base_path,
            model_config_metrics_dict=self.model_config_metrics_dict,
            config_params_to_label=config_params_to_label,
            save_path=save_path,
            subplots_kwargs=subplots_kwargs,
            seaborn_kwargs=seaborn_kwargs,
            colormap=colormap,
        )
        return comparer.generate_plot()
        
    @staticmethod
    def _get_available_paths_with_models(
        base_path: str
    ) -> List[str]:
        paths_with_models = []
        for root, dirs, files in os.walk(base_path):
            if "config.yaml" in files and \
                "Metrics_and_losses.csv" in files and \
                "Model" in dirs:
                paths_with_models.append(root)
        return paths_with_models
    
    def _get_model_config_metrics_dict(
        self,
        model_paths: List[str]
    ) -> Dict:
        _model_config_metrics_dict = dict()
        for path in model_paths:
            _model_config_metrics_dict[path] = {
                "config": self._load_config(os.path.join(path, "config.yaml")),
                "metrics": self._load_metrics(os.path.join(path, "Metrics_and_losses.csv")),
                "model": None
            }
        
        for path in model_paths:
            _model_config_metrics_dict[path]["model"] = self._load_model(
                os.path.join(path, "Model"),
                _model_config_metrics_dict[path]["config"]["metrics"]
            )

        return _model_config_metrics_dict
                
    @staticmethod    
    def _load_model(
        path: str,
        model_metrics: List = None,
    ):
        model_metrics = [] if model_metrics is None else model_metrics
        model_metrics = [model_metrics] if type(model_metrics) not in (list, tuple) else model_metrics

        return load_model(
            path,
            custom_objects={_metric: metrics_dict[_metric] for _metric in model_metrics}
        )
    
    @staticmethod
    def _load_config(
        path: str
    ) -> Dict:
        with open(path) as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def _load_metrics(
        path: str
    ) -> pd.DataFrame:
        return pd.read_csv(path)


__all__ = [
    "ModelComparer",
]

if __name__ == "__main__":
    pass
    # model_comparer = ModelComparer(base_path=os.path.join("..", "..", "Models", "CDT_1D"))

