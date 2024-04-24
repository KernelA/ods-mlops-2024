import json
import pathlib

import hydra
import matplotlib as mpl
import numpy as np
import polars as pl
import seaborn as sea
from hydra.core.config_store import ConfigStore
from sklearn import compose, model_selection, pipeline, preprocessing

mpl.use("module://mplcairo.base")

from ods_mlops.configs import PreprocessConfig
from ods_mlops.sk_ext import BooleanEncoder

cs = ConfigStore.instance()
cs.store(name="base_preprocess", node=PreprocessConfig)


@hydra.main(config_path="configs", config_name="preprocess", version_base="1.3")
def main(config: PreprocessConfig):
    data = (
        pl.scan_parquet(
            config.input_data,
        )
        .filter(pl.col(config.data_info.target_col).is_not_null())
        .select(
            config.data_info.bin_cols
            + config.data_info.cat_cols
            + [config.data_info.target_col]
            + config.data_info.numeric_cols,
        )
        .with_row_index()
        .collect()
    )

    encoder = preprocessing.LabelEncoder()
    target = encoder.fit_transform(data.get_column(config.data_info.target_col).to_numpy())

    out_dir = pathlib.Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "class_labels.json").open("w", encoding="utf-8") as f:
        json.dump({i: cls for i, cls in enumerate(encoder.classes_)}, f)

    train_index, test_index, train_target, test_target = model_selection.train_test_split(
        data.get_column("index").to_numpy(),
        target,
        stratify=target,
        random_state=config.split_info.seed,
    )

    transforms = []

    if config.data_info.numeric_cols:
        transforms.append(
            (
                "numeric",
                preprocessing.StandardScaler(),
                [data.columns.index(col) for col in config.data_info.numeric_cols],
            ),
        )

    if config.data_info.cat_cols:
        transforms.append(
            (
                "cat",
                preprocessing.OneHotEncoder(),
                [data.columns.index(col) for col in config.data_info.cat_cols],
            ),
        )

    if config.data_info.bin_cols:
        transforms.append(
            (
                "bool",
                BooleanEncoder(),
                [data.columns.index(col) for col in config.data_info.bin_cols],
            )
        )

    tr = compose.ColumnTransformer(transforms)

    train = data.filter(pl.col("index").is_in(train_index))
    test = data.filter(pl.col("index").is_in(test_index))

    train_features = tr.fit_transform(train)
    test_features = tr.transform(test)

    np.save(out_dir / "train_features.npy", train_features)
    np.save(out_dir / "test_features.npy", test_features)
    np.save(out_dir / "train_target.npy", train_target)
    np.save(out_dir / "test_target.npy", test_target)

    target_info = (
        pl.concat(
            [
                pl.DataFrame({"target": train_target}).with_columns(pl.lit("train").alias("split")),
                pl.DataFrame({"target": test_target}).with_columns(pl.lit("test").alias("split")),
            ]
        )
        .group_by("split", "target")
        .len()
    )

    ax = sea.barplot(target_info, x="target", y="len", hue="split")
    fig = ax.get_figure()
    fig.savefig(str(out_dir / "target_distr.png"))


if __name__ == "__main__":
    main()
