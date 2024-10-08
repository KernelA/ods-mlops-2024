import pathlib
from typing import List

import fsspec
import hydra
import polars as pl
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from ods_mlops.sms_cls.extract_data import extract_data


def split_data(data: pl.DataFrame, train_size: float, seed: int):
    target = data.get_column("target").to_physical().to_numpy()
    data = data.with_row_index()
    train, test = train_test_split(
        data.get_column("index").to_numpy(),
        stratify=target,
        train_size=train_size,
        random_state=seed,
    )
    return data.filter(pl.col("index").is_in(train)).drop("index"), data.filter(
        pl.col("index").is_in(test)
    ).drop("index")


@hydra.main(config_path="configs_sms", config_name="train", version_base="1.3")
def main(config: DictConfig):
    out_dir = pathlib.Path(config.data.split_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fs: fsspec.AbstractFileSystem = fsspec.url_to_fs(config.data.fsspec_path)[0]

    messages = []

    for file in fs.glob("/sentiment labelled sentences/**/*.txt"):
        with fs.open(file, "rt", encoding="utf-8") as f:
            messages.append(extract_data(f, config.data.class_mapping))

    assert len(messages) > 0

    messages = pl.concat(messages, how="vertical")

    train, not_test = split_data(messages, config.split_info.train_size, config.split_info.seed)
    train.write_parquet(out_dir / "train.parquet")
    del train

    val, test = split_data(not_test, config.split_info.val_size, config.split_info.seed)

    test.write_parquet(out_dir / "test.parquet")
    val.write_parquet(out_dir / "val.parquet")


if __name__ == "__main__":
    main()
