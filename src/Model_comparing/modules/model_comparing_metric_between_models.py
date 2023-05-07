import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import seaborn as sns

from datetime import datetime
from typing import Dict, List, Union

from .model_comparing_helper import dict_to_label_string, legend_adjust


class ModelComparerMetricBetweenModels:
    defaulf_kwargs_subplot = {
        "figsize": (25, 15)
    }
    defaulf_kwargs_seaborn = {}
    default_colormap = mpl.colormaps["tab20"]

    def __init__(
        self,
        metric: str,
        base_path: str,
        model_config_metrics_dict: Dict,
        config_params_to_label: List = None,
        save_path: str = None,
        subplots_kwargs: dict = None,
        seaborn_kwargs: dict = None,
        colormap: Union[str, mpl.colors.ListedColormap] = None
    ) -> plt.figure:
        self.metric = metric
        self.base_path = base_path
        self.model_config_metrics_dict = model_config_metrics_dict

        self.config_params_to_label = config_params_to_label

        self.save_path = self._generate_plot_save_path(
            plot_name = metric + "_compare_metric_between_models",
            label_params = config_params_to_label,
            type_for_dirname = metric
        ) if save_path is None else save_path

        self.subplots_kwargs = self.__class__.defaulf_kwargs_subplot.copy() if subplots_kwargs is None else subplots_kwargs
        self.seaborn_kwargs = self.__class__.defaulf_kwargs_seaborn.copy() if seaborn_kwargs is None else seaborn_kwargs
        self.colormap = self.__class__.default_colormap if colormap is None else \
            mpl.colormaps[colormap] if type(colormap) == str else \
            colormap
    
    def generate_plot(self) -> plt.figure:
        fig, ax = plt.subplots(**self.subplots_kwargs)

        self._generate_plot_on_ax(
            metric=self.metric,
            ax=ax,
        )
        legend_adjust(
            _ax = ax,
            _loc = "upper left",
            _bbox_to_anchor = (1, 1),
            _cmap = self.colormap,
        )
        fig.suptitle(
            f"Metric '{self.metric}' of {len(self.model_config_metrics_dict.keys())} models\n"
            f"with differance in parameters described in legend\n"
            f"from '{self.base_path}'"
        )
        plt.grid(
            figure=fig,
            alpha=0.6
        )

        fig.savefig(self.save_path, bbox_inches="tight")
        plt.close(fig)
        return fig

    def _generate_plot_save_path(
        self,
        plot_name: str,
        label_params: List,
        type_for_dirname: str,
    ) -> str:
        path_dir = os.path.join(
            "Plots",
            "Model_comparing",
            self.base_path.replace(os.sep, "_"),
            type_for_dirname
        )
        os.makedirs(path_dir, exist_ok=True)

        path_filename = plot_name + \
            "_" + "_".join(label_params) + \
            "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".png"
        return os.path.join(path_dir, path_filename)
    
    def _generate_plot_on_ax(
        self,
        metric: str,
        ax: plt.Axes,
    ):
        for k in self.model_config_metrics_dict.keys():
            _config = self.model_config_metrics_dict[k]["config"]
            _metric_df = self.model_config_metrics_dict[k]["metrics"][[metric]]
            _label = dict_to_label_string(
                {_param: str(_config[_param]) for _param in self.config_params_to_label},
                _model_path=k
            )
            
            sns.lineplot(
                data=_metric_df,
                x=_metric_df.index,
                y=metric,
                ax=ax,
                label=_label,
                **self.seaborn_kwargs
            )
        ax.locator_params(nbins=20, axis='x')
        ax.locator_params(nbins=20, axis='y')

            


__all__ = [
    "ModelComparerMetricBetweenModels"
]
