import polars as pl
import seaborn as sea


def heat_map(data: pl.DataFrame, ax=None):
    y_labels = data.get_column(data.columns[0]).to_list()
    x_labels = data.columns[1:]

    heat_data = data.select(pl.col("*").exclude(data.columns[0])).to_numpy()
    return sea.heatmap(
        heat_data,
        xticklabels=x_labels,
        yticklabels=y_labels,
        ax=ax,
        cmap=sea.color_palette("coolwarm", as_cmap=True),
    )
