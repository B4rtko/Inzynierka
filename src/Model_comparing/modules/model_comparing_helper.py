import matplotlib as mpl
import matplotlib.pyplot as plt

from typing import Dict, Tuple

            
def dict_to_label_string(
    _dict: Dict,
    _model_path: str = None,
):
    return ",\n".join([f"{_key}={_value}" for (_key, _value) in _dict.items()]) + f"\nmodel_path={_model_path}"


def legend_adjust(
    _ax: plt.Axes,
    _loc: str = "upper left",
    _bbox_to_anchor: Tuple[int, int] = (1, 1),
    _cmap: mpl.colors.ListedColormap = mpl.colormaps["tab20"]
):
    lines = _ax.lines
    colors = _cmap(range(len(lines)))
    for line, c in zip(lines, colors):
        line.set_color(c)
    
    _ax.legend(loc=_loc, bbox_to_anchor=_bbox_to_anchor)


__all__ = [
    "dict_to_label_string",
    "legend_adjust",
]