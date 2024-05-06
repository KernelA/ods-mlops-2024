import matplotlib.pyplot as plt
import polars as pl
from matplotlib.axes import Axes


def scatter_plot(data: pl.DataFrame, x_col: str, y_col: str, label_col: str):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for label, group in data.group_by(label_col):
        ax.plot(
            group.get_column(x_col),
            group.get_column(y_col),
            marker=".",
            label=label,
            linestyle="None",
            markersize=2,
            alpha=0.2,
        )

    ax.legend()

    return fig
