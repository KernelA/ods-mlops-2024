import pathlib
from abc import ABC, abstractmethod
from typing import Any, Literal, Tuple

import numpy as np
import polars as pl
from numpy import load, save
from scipy.sparse import load_npz, save_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


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


class Vectorizer:
    def __init__(self, stat_type: Literal["count", "tfidf"]):
        self._vectorizer = TfidfVectorizer() if stat_type == "tfidf" else CountVectorizer()

    def _vec_train(self, train: pl.DataFrame):
        return self._vectorizer.fit_transform((row["text"] for row in train.iter_rows(named=True)))

    def _vec_other(self, other: pl.DataFrame):
        return self._vectorizer.transform((row["text"] for row in other.iter_rows(named=True)))

    def vectorize_train(self, train: pl.DataFrame, out_dir: pathlib.Path):
        features = self._vec_train(train)
        save_npz(out_dir / "features.npz", features)
        train.select("target").write_parquet(out_dir / "target.parquet")

    def _vectorize_other(self, other: pl.DataFrame, out_dir: pathlib.Path):
        features = self._vec_other(other)
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


class SVDVectorizer(Vectorizer):
    def __init__(self, n_components: int, n_iter: int, random_state: int):
        super().__init__("tfidf")
        self._svd = TruncatedSVD(
            n_components=n_components, n_iter=n_iter, random_state=random_state
        )

    def vectorize_train(self, train: pl.DataFrame, out_dir: pathlib.Path):
        features = self._svd.fit_transform(super()._vec_train(train))
        save(out_dir / "features.npy", features)
        train.select("target").write_parquet(out_dir / "target.parquet")

    def _vectorize_other(self, other: pl.DataFrame, out_dir: pathlib.Path):
        features = self._svd.transform(super()._vec_other(other))
        save(out_dir / "features.npy", features)
        other.select("target").write_parquet(out_dir / "target.parquet")

    def load_data(self, data_dir: pathlib.Path):
        features = load(data_dir / "features.npy")
        target = pl.read_parquet(data_dir / "target.parquet")
        return features, target.get_column("target").to_numpy()
