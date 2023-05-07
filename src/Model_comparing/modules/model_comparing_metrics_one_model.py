import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

from datetime import datetime
from typing import Dict, List, Union

from .model_comparing_helper import dict_to_label_string, legend_adjust


class ModelComparerMetricsOneModel:
    defaulf_kwargs_subplot = {
        "figsize": (25, 15)
    }
    defaulf_kwargs_seaborn = {}
    default_colormap = mpl.colormaps["tab20"]

    def __init__(
        self,
        base_path: str,
        model_config_metrics_dict: Dict,
        metric_1: str = "loss",
        metric_2: str = "f1_m",
        config_params_to_title: List = None,
        subplots_kwargs: dict = None,
        seaborn_kwargs: dict = None,
        colormap: Union[str, mpl.colors.ListedColormap] = None
    ) -> plt.figure:
        self.metric_1 = metric_1
        self.metric_2 = metric_2
        
        self.base_path = base_path
        self.model_config_metrics_dict = model_config_metrics_dict

        self.config_params_to_title = config_params_to_title

        self.subplots_kwargs = self.__class__.defaulf_kwargs_subplot.copy() if subplots_kwargs is None else subplots_kwargs
        self.seaborn_kwargs = self.__class__.defaulf_kwargs_seaborn.copy() if seaborn_kwargs is None else seaborn_kwargs
        self.colormap = self.__class__.default_colormap if colormap is None else \
            mpl.colormaps[colormap] if type(colormap) == str else \
            colormap
    
    def generate_plot(self) -> plt.figure:
        for k in self.model_config_metrics_dict.keys():
            _config = self.model_config_metrics_dict[k]["config"]
            _metric_df = self.model_config_metrics_dict[k]["metrics"]
            _title = dict_to_label_string(
                {_param: str(_config[_param]) for _param in self.config_params_to_title},
                _model_path=k
            )
            _save_path = self._generate_plot_save_path(
                plot_name = os.path.basename(k),
                type_for_dirname = f"{self.metric_1}_vs_{self.metric_2}_one_model"
            )

            fig, ax = plt.subplots(**self.subplots_kwargs)

            self._generate_plot_on_ax(
                metrics_df=_metric_df,
                ax=ax,
            )
            # legend_adjust(
            #     _ax = ax,
            #     _loc = "upper left",
            #     _bbox_to_anchor = (1, 1),
            #     _cmap = self.colormap,
            # )
            fig.suptitle(
                f"Metric values comparision of model with parameters:\n"
                f"{_title}."
            )
            plt.grid(
                figure=fig,
                alpha=0.6
            )

            fig.savefig(_save_path, bbox_inches="tight")
            plt.close(fig)
    
    def _generate_plot_on_ax(
        self,
        metrics_df: pd.DataFrame,
        ax: plt.Axes,
    ):
        ax_2 = ax.twinx()

        df_ax_1 = metrics_df.iloc[:, metrics_df.columns.str.endswith(self.metric_1)]
        df_ax_2 = metrics_df.iloc[:, metrics_df.columns.str.endswith(self.metric_2)]
        
        palette_ax_1 = self.colormap(range(len(df_ax_1.columns)))
        palette_ax_2 = self.colormap(
            range(
                len(df_ax_1.columns),
                len(df_ax_1.columns) + len(df_ax_2.columns)
            )
        )

        df_ax_1_melted = pd.melt(
            df_ax_1,
            ignore_index=False,
            value_name=self.metric_1 + "_"
        )
        df_ax_2_melted = pd.melt(
            df_ax_2,
            ignore_index=False,
            value_name=self.metric_2 + "_"
        )

        sns.lineplot(
            data=df_ax_1_melted,
            x=df_ax_1_melted.index,
            y=self.metric_1 + "_",
            ax=ax,
            hue="variable",
            palette=palette_ax_1,
            **self.seaborn_kwargs
        )

        sns.lineplot(
            data=df_ax_2_melted,
            x=df_ax_2_melted.index,
            y=self.metric_2 + "_",
            ax=ax_2,
            hue="variable",
            palette=palette_ax_2,
            **self.seaborn_kwargs
        )

        ax.locator_params(nbins=20, axis='x')
        ax.locator_params(nbins=20, axis='y')
        ax_2.locator_params(nbins=20, axis='y')


    def _generate_plot_save_path(
        self,
        plot_name: str,
        type_for_dirname: str,
    ) -> str:
        path_dir = os.path.join(
            "Plots",
            "Metrics_comparing",
            self.base_path.replace(os.sep, "_"),
            type_for_dirname
        )
        os.makedirs(path_dir, exist_ok=True)

        path_filename = plot_name + \
            "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".png"
        return os.path.join(path_dir, path_filename)


__all__ = [
    "ModelComparerMetricsOneModel"
]
