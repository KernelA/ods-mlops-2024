import pathlib

import hydra
import polars as pl
from omegaconf import DictConfig

from ods_mlops.sms_cls.vectorizer import BaseVectorizer


@hydra.main(config_path="configs_sms", config_name="train", version_base="1.3")
def main(config: DictConfig):
    data_dir = pathlib.Path(config.data.split_dir)

    vect_dir = pathlib.Path(config.data.vectorized_dir)
    vect_dir.mkdir(parents=True, exist_ok=True)

    vectorizer: BaseVectorizer = hydra.utils.instantiate(config.vectorizer)

    for split_dir_name in ("train", "val", "test"):
        out_dir = vect_dir / split_dir_name
        out_dir.mkdir(exist_ok=True, parents=True)

        data = pl.read_parquet(data_dir / f"{split_dir_name}.parquet")

        if split_dir_name == "train":
            vectorizer.vectorize_train(data, out_dir)
        else:
            call = getattr(vectorizer, f"vectorize_{split_dir_name}")
            call(data, out_dir)


if __name__ == "__main__":
    main()
