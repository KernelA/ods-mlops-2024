import json
import pathlib

from sklearnex import patch_sklearn

patch_sklearn()

import hydra
import matplotlib as mpl
import numpy as np
import polars as pl
from hydra.core.config_store import ConfigStore
from sklearn.base import ClassifierMixin
from sklearn.metrics import confusion_matrix

mpl.use("module://mplcairo.base")

import seaborn as sea

from ods_mlops.configs import TrainConfig

cs = ConfigStore.instance()
cs.store(name="base_train", node=TrainConfig)


def load_data(split: str, in_dir: pathlib.Path):
    return np.load(in_dir / f"{split}_features.npy"), np.load(in_dir / f"{split}_target.npy")


@hydra.main(config_name="train", config_path="configs", version_base="1.3")
def main(config: TrainConfig):
    model: ClassifierMixin = hydra.utils.instantiate(config.model)

    in_dir = pathlib.Path(config.input_dir)

    train_features, train_target = load_data("train", in_dir)

    model.fit(train_features, train_target)

    del train_features
    del train_target

    test_features, test_target = load_data("test", in_dir)

    predicted_target = model.predict(test_features)

    with (in_dir / "class_labels.json").open("rb") as f:
        class_labels = list(map(lambda x: x[1], sorted(json.load(f).items())))

    matrix = confusion_matrix(test_target, predicted_target, normalize="all")

    data = pl.from_numpy(matrix, class_labels)
    data = pl.concat([pl.DataFrame({"class": data.columns}), data], how="horizontal")

    out_dir = pathlib.Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data.write_csv(out_dir / "conf_matrix.csv")

    ax = sea.heatmap(
        matrix,
        xticklabels=class_labels,
        yticklabels=class_labels,
        annot=True,
        square=True,
        cmap=sea.color_palette("coolwarm", as_cmap=True),
    )
    ax.get_figure().savefig(out_dir / "conf_matrix.png")


if __name__ == "__main__":
    main()
