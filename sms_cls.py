import pathlib

import hydra
import polars as pl
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix

from ods_mlops.sms_cls.model import BaseModel


@hydra.main(config_path="configs_sms", config_name="train", version_base="1.3")
def main(config: DictConfig):
    vect_dir = pathlib.Path(config.data.vectorized_dir)
    model: BaseModel = hydra.utils.instantiate(config.model)

    model.fit(vect_dir / "train")
    val_predict = model.predict(vect_dir / "val")

    conf_matrix = confusion_matrix(
        val_predict.true_labels,
        val_predict.predicted_lables,
        normalize="all",
    )

    class_labels = list(
        map(lambda x: x[0], sorted(config.data.class_mapping.items(), key=lambda x: x[1]))
    )

    data = pl.from_numpy(
        conf_matrix,
        class_labels,
    )
    data = pl.concat([pl.DataFrame({"class": data.columns}), data], how="horizontal")

    metric_dir = pathlib.Path(config.data.metrics_dir)
    metric_dir.mkdir(parents=True, exist_ok=True)
    data.write_csv(metric_dir / "conf_matrix.csv")

    # ax = sea.heatmap(
    #     matrix,
    #     xticklabels=class_labels,
    #     yticklabels=class_labels,
    #     annot=True,
    #     square=True,
    #     cmap=sea.color_palette("coolwarm", as_cmap=True),
    # )
    # ax.get_figure().savefig(out_dir / "conf_matrix.png")


if __name__ == "__main__":
    main()
