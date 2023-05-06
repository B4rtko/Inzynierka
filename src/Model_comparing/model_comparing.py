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


class ModelComparer:
    def __init__(
        self,
        base_path: str = os.path.join("Models"),
        
    ) -> None:
        self.base_path = base_path

        self.model_paths = self._get_available_paths_with_models(self.base_path)
        self.model_config_metrics_dict = dict()
        self.defaulf_kwargs_subplot = {
            "figsize": (25, 15)
        }
        self.defaulf_kwargs_seaborn = dict()
        
    def populate_model_config_metrics_dict(self):
        self.model_config_metrics_dict = self._get_model_config_metrics_dict(self.model_paths)
    
    def compare_metric_between_models(
        self,
        metric: str,
        config_params_to_label: List = None,
        save_path: str = None,
        subplots_kwargs: dict = None,
        seaborn_kwargs: dict = None,
    ) -> plt.figure:
        subplots_kwargs = self.defaulf_kwargs_subplot if subplots_kwargs is None else subplots_kwargs
        seaborn_kwargs = self.defaulf_kwargs_seaborn if seaborn_kwargs is None else seaborn_kwargs
        save_path = self._generate_plot_save_path(metric, config_params_to_label) if save_path is None else save_path
        
        fig, ax = plt.subplots(**subplots_kwargs)

        self._compare_metric_plot_on_ax(
            metric=metric,
            config_params_to_label=config_params_to_label,
            ax=ax,
            seaborn_kwargs=seaborn_kwargs
        )
        
        ax = self._legend_adjust(
            _ax = ax,
            _loc = "upper left",
            _bbox_to_anchor = (1, 1),
            _cmap = mpl.colormaps["tab20"],
        )
                
        fig.suptitle(
            f"Metric '{metric}' of {len(self.model_config_metrics_dict.keys())} models\n"
            f"with differance in parameters described in legend\n"
            f"from '{self.base_path}'"
        )
        
        fig.savefig(save_path, bbox_inches="tight")
    
    @staticmethod
    def _legend_adjust(
        _ax: plt.Axes,
        _loc: str = "upper left",
        _bbox_to_anchor: Tuple[int, int] = (1, 1),
        _cmap = mpl.colormaps["tab20"]
    ):
        lines = _ax.lines
        colors = _cmap(range(len(lines)))
        for line, c in zip(lines, colors):
            line.set_color(c)
        
        _ax.legend(loc=_loc, bbox_to_anchor=_bbox_to_anchor)

        
    
    def _compare_metric_plot_on_ax(
        self,
        metric: str,
        config_params_to_label: List,
        ax: plt.axis,
        seaborn_kwargs: dict,
    ):
        for k in self.model_config_metrics_dict.keys():
            _config = self.model_config_metrics_dict[k]["config"]
            _metric_df = self.model_config_metrics_dict[k]["metrics"][[metric]]
            _label = self._dict_to_label_string(
                {_param: str(_config[_param]) for _param in config_params_to_label},
                _model_path=k
            )
            
            sns.lineplot(
                data=_metric_df,
                x=_metric_df.index,
                y=metric, ax=ax,
                label=_label,
                **seaborn_kwargs
            )
    
    def _generate_plot_save_path(
        self,
        metric: str,
        label_params: List
    ) -> str:
        path_dir = os.path.join(
            "Plots",
            "Model_comparing",
            self.base_path.replace(os.pathsep, "_"),
            metric
        )
        os.makedirs(path_dir, exist_ok=True)

        path_filename = metric + "_" + "_".join(label_params) + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".png"
        return os.path.join(path_dir, path_filename)
            
            
    @staticmethod
    def _dict_to_label_string(
        _dict: Dict,
        _model_path: str = None,
    ):
        return ",\n".join([f"{_key}={_value}" for (_key, _value) in _dict.items()]) + f"\nmodel_path={_model_path}"
    
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

