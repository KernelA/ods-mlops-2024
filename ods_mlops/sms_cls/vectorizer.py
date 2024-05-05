import pathlib
from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np
import polars as pl
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


class BaseVectorizer(ABC):
    @abstractmethod
    def vectorize_train(self, train: pl.DataFrame, out_dir: pathlib.Path):
        pass

    @abstractmethod
    def vectorize_test(self, train: pl.DataFrame, out_dir: pathlib.Path):
        pass

    @abstractmethod
    def vectorize_val(self, train: pl.DataFrame, out_dir: pathlib.Path):
        pass

    @abstractmethod
    def load_data(self, data_dir: pathlib.Path) -> Tuple[Any, np.ndarray]:
        pass

    @abstractmethod
    def dump(self, out_dir: pathlib.Path):
        pass


class TfIdfVectorizer:
    def __init__(self):
        self._vectorizer = TfidfVectorizer()

    def vectorize_train(self, train: pl.DataFrame, out_dir: pathlib.Path):
        features = self._vectorizer.fit_transform(
            (row["text"] for row in train.iter_rows(named=True))
        )
        save_npz(out_dir / "features.npz", features)
        train.select("target").write_parquet(out_dir / "target.parquet")

    def _vectorize_other(self, other: pl.DataFrame, out_dir: pathlib.Path):
        features = self._vectorizer.transform((row["text"] for row in other.iter_rows(named=True)))
        save_npz(out_dir / "features.npz", features)
        other.select("target").write_parquet(out_dir / "target.parquet")

    def vectorize_test(self, test: pl.DataFrame, out_dir: pathlib.Path):
        self._vectorize_other(test, out_dir)

    def vectorize_val(self, val: pl.DataFrame, out_dir: pathlib.Path):
        self._vectorize_other(val, out_dir)

    def load_data(self, data_dir: pathlib.Path):
        features = load_npz(data_dir / "features.npz")
        target = pl.read_parquet(data_dir / "target.parquet")
        return features, target.get_column("target").to_numpy()

    def dump(self, out_dir: pathlib.Path):
        pass
