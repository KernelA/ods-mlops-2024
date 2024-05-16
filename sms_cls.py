import json
import pathlib

import sklearnex

sklearnex.patch_sklearn()

import hydra
import polars as pl
import seaborn as sea
from flatten_dict import flatten
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import confusion_matrix, roc_auc_score

from ods_mlops.exp_loggers.base_logger import BaseExpLogger
from ods_mlops.sms_cls.model import BaseModel


@hydra.main(config_path="configs_sms", config_name="train", version_base="1.3")
def main(config: DictConfig):
    exp_logger: BaseExpLogger = hydra.utils.instantiate(config.exp_logger)
    vect_dir = pathlib.Path(config.data.vectorized_dir)
    model: BaseModel = hydra.utils.instantiate(config.model.cls)

    model.fit(vect_dir / "train")

    exp_logger.init()

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

        exp_logger.log_params(
            flatten(
                OmegaConf.to_container(config, resolve=True, enum_to_str=True),
                reducer="dot",
                enumerate_types=(list,),
            )
        )

        fig = plt.figure(figsize=(10, 10), dpi=150)
        ax = fig.add_subplot(111)
        sea.heatmap(
            matrix,
            xticklabels=class_labels,
            yticklabels=class_labels,
            annot=True,
            square=True,
            cmap=sea.color_palette("coolwarm", as_cmap=True),
            ax=ax,
        )
        exp_logger.log_figure(ax.get_figure(), f"{split_type}_conf_matrix.png")

        with open(metric_dir / "roc_auc.json", "w", encoding="utf-8") as f:
            metric = {f"{split_type}_roc_auc": roc_auc}
            exp_logger.log_metric(f"{split_type}_roc_auc", metric[f"{split_type}_roc_auc"])
            json.dump(metric, f)


if __name__ == "__main__":
    main()
