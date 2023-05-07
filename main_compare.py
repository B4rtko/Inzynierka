import os

from typing import Dict, List, Tuple, Union

from src.Model_comparing import ModelComparer


def compare(
    base_path: str,
    types: Union[str, List] = "all",
    config_params: List[str] = None,
    **kwargs
):
    model_comparer = ModelComparer(base_path=base_path)
    model_comparer.populate_model_config_metrics_dict()
    config_params = [
        "convolution_layers_count", "learning_rate", "dense_layers_units"
    ] if config_params is None else config_params
    
    types = list(compare_types_and_methods.keys()) if types == "all" else \
        [types] if type(types) == str else \
        types
    
    for t in types:
        _kwargs = kwargs[t] if t in kwargs.keys() else {}
        
        compare_types_and_methods[t](
            model_comparer,
            config_params=config_params,
            **_kwargs
        )
    


def _compare_metric_between_models(
    model_comparer: ModelComparer,
    metrics: str = "all",
    config_params: List[str] = None,
    subplots_kwargs: Dict = None
):
    metrics = comparable_metrics_list if metrics == "all" else metrics
    
    if type(metrics) == str:
        model_comparer.compare_metric_between_models(
            metrics,
            config_params_to_label=config_params,
            subplots_kwargs = subplots_kwargs
        )
    
    else:
        for _metric in metrics:
            _compare_metric_between_models(
                model_comparer = model_comparer,
                metrics = _metric,
                config_params = config_params,
                subplots_kwargs = subplots_kwargs,
            )


def _compare_metrics_one_model(
    model_comparer: ModelComparer,
    metric_pairs: Union[Tuple[str, str], Tuple[Tuple[str, str], ...]],
    config_params: List[str],
    subplots_kwargs: Dict = None
):
    if len(metric_pairs) == 2 and type(metric_pairs[0]) == str and type(metric_pairs[1]) == str:
        metric_1, metric_2 = metric_pairs

        model_comparer.compare_metrics_of_one_model(
            metric_1 = metric_1,
            metric_2 = metric_2,
            config_params_to_title=config_params,
            subplots_kwargs = subplots_kwargs
        )
    else:
        for _metric_pair in metric_pairs:
            _compare_metrics_one_model(
                model_comparer=model_comparer,
                metric_pairs=_metric_pair,
                config_params=config_params,
                subplots_kwargs=subplots_kwargs,
            )


compare_types_and_methods = {
    "metric_between_models": _compare_metric_between_models,
    "metrics_one_model": _compare_metrics_one_model,
}


comparable_metrics_list = [
    "loss",
    "val_loss",
    "f1_m",
    "val_f1_m",
]


if __name__ == "__main__":
    model_dirs = \
    os.listdir(os.path.join("Models", "CDT_1D"))
    ["Apple_5"]
    
    types = \
    "all"
    "metric_between_models"
    "metrics_one_model"
    
    config_params = \
    None
    
    kwargs = \
    {
        "metrics_one_model": {
            "metric_pairs": ("loss", "f1_m")
        }
    }

    for d in model_dirs:
        _base_path = os.path.join("Models", "CDT_1D", d, "Completed")
        compare(
            base_path=_base_path,
            types=types,
            config_params=config_params,
            **kwargs
        )



