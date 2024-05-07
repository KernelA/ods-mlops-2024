import json
import pathlib

import sklearnex

sklearnex.patch_sklearn()

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix, roc_auc_score

from ods_mlops.sms_cls.model import BaseModel


@hydra.main(config_path="configs_sms", config_name="train", version_base="1.3")
def main(config: DictConfig):
    vect_dir = pathlib.Path(config.data.vectorized_dir)
    model: BaseModel = hydra.utils.instantiate(config.model.cls)

    model.fit(vect_dir / "train")

    for split_type in ("train", "test", "val"):
        predict = model.predict(vect_dir / split_type)

        class_labels = list(
            map(lambda x: x[0], sorted(config.data.class_mapping.items(), key=lambda x: x[1]))
        )

        metric_dir = pathlib.Path(config.data.metrics_dir) / split_type
        metric_dir.mkdir(parents=True, exist_ok=True)

        matrix = confusion_matrix(
            predict.true_labels,
            predict.predicted_labels,
        )

        data = pl.from_numpy(matrix, class_labels)
        data = pl.concat([pl.DataFrame({"class": data.columns}), data], how="horizontal")

        inv_class_labels = {v: k for k, v in config.data.class_mapping.items()}

        data = pl.DataFrame({"actual": predict.true_labels, "predicted": predict.predicted_labels})
        data = data.with_columns(
            pl.col("actual").replace(inv_class_labels, return_dtype=pl.String),
            pl.col("predicted").replace(inv_class_labels, return_dtype=pl.String),
        )

        data.write_csv(metric_dir / "conf_matrix.csv")

        roc_auc = roc_auc_score(predict.true_labels, predict.predicted_scores)

        with open(metric_dir / "roc_auc.json", "w", encoding="utf-8") as f:
            json.dump({f"{split_type}_roc_auc": roc_auc}, f)


if __name__ == "__main__":
    main()
