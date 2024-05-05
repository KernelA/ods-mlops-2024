import json
import pathlib

import hydra
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

        conf_matrix = confusion_matrix(
            predict.true_labels,
            predict.predicted_lables,
            normalize="all",
        )

        class_labels = list(
            map(lambda x: x[0], sorted(config.data.class_mapping.items(), key=lambda x: x[1]))
        )

        data = pl.from_numpy(
            conf_matrix,
            class_labels,
        )

        metric_dir = pathlib.Path(config.data.metrics_dir) / split_type
        metric_dir.mkdir(parents=True, exist_ok=True)

        roc_auc = roc_auc_score(predict.true_labels, predict.predicted_lables)

        with open(metric_dir / "roc_auc.json", "w", encoding="utf-8") as f:
            json.dump({f"{split_type}_roc_auc": roc_auc}, f)

        data.write_json(metric_dir / "conf_matrix.json", row_oriented=True)


if __name__ == "__main__":
    main()
