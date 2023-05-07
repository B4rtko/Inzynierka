import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

from tensorflow.keras.models import load_model
from typing import List, Dict, Tuple, Union
from datetime import datetime

from src.Model_training.cdt_1d import metrics_dict
from src.Model_comparing.modules import ModelComparerMetricBetweenModels


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
    
    # def compare_metrics_of_one_model(self):
    #     if subplots_kwargs is None:
    #         subplots_kwargs = self.defaulf_kwargs_subplot.copy()
    #         _figsize = seaborn_kwargs["figsize"]
    #         seaborn_kwargs["figsize"] = (_figsize[0], _figsize[1]*2)

    #     seaborn_kwargs = self.defaulf_kwargs_seaborn if seaborn_kwargs is None else seaborn_kwargs

    #     save_path = self._generate_plot_save_path_compare_metrics_of_one_model(
    #         plot_name = "compare_metrics_of_one_model",
    #         label_params = config_params_to_label,
    #         type_for_dirname = metric
    #     ) if save_path is None else save_path
        
    #     fig, ax = plt.subplots((2, 1), **subplots_kwargs)

    #     self._compare_metric_plot_on_ax(
    #         metric=metric,
    #         config_params_to_label=config_params_to_label,
    #         ax=ax,
    #         seaborn_kwargs=seaborn_kwargs
    #     )
        
    #     ax = self._legend_adjust(
    #         _ax = ax,
    #         _loc = "upper left",
    #         _bbox_to_anchor = (1, 1),
    #         _cmap = mpl.colormaps["tab20"],
    #     )
                
    #     fig.suptitle(
    #         f"Metric '{metric}' of {len(self.model_config_metrics_dict.keys())} models\n"
    #         f"with differance in parameters described in legend\n"
    #         f"from '{self.base_path}'"
    #     )
        
    #     fig.savefig(save_path, bbox_inches="tight")


    # def _generate_plot_save_path_compare_metrics_of_one_model(
    #     self,
    #     plot_name: str,
    #     label_params: List,
    #     type_for_dirname: str,
    # ) -> str:
    #     path_dir = os.path.join(
    #         "Plots",
    #         "Metrics_of_one_model_comparing",
    #         self.base_path.replace(os.pathsep, "_"),
    #         type_for_dirname
    #     )
    #     os.makedirs(path_dir, exist_ok=True)

    #     path_filename = plot_name + \
    #         "_" + "_".join(label_params) + \
    #         "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".png"
    #     return os.path.join(path_dir, path_filename)
    
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

