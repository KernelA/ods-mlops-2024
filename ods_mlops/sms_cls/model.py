import pathlib
import pickle
from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np
from sklearn.base import ClassifierMixin

from .vectorizer import BaseVectorizer


class Predict(NamedTuple):
    true_labels: np.ndarray
    predicted_lables: np.ndarray


class BaseModel(ABC):
    def __init__(self, vectorizer: BaseVectorizer) -> None:
        super().__init__()
        self.vectorizer = vectorizer

    @abstractmethod
    def fit(self, data_dir: pathlib.Path):
        pass

    @abstractmethod
    def save(self, data_dir: pathlib.Path):
        pass

    @staticmethod
    def load(data_dir: pathlib.Path) -> "BaseModel":
        pass

    @abstractmethod
    def predict(self, data_dir: pathlib.Path) -> Predict:
        pass


class SkCls(BaseModel):
    def __init__(self, vectorizer: BaseVectorizer, estimator: ClassifierMixin):
        super().__init__(vectorizer=vectorizer)
        self.estimator = estimator

    def fit(self, data_dir: pathlib.Path):
        features, target = self.vectorizer.load_data(data_dir)
        self.estimator.fit(features, target)

    def save(self, data_dir: pathlib.Path):
        with open(data_dir / "model.pickle", "wb") as f:
            pickle.dump(self.estimator, f)

        with open(data_dir / "vectorizer.pickle", "wb") as f:
            pickle.dump(self.estimator, f)

    def predict(self, data_dir: pathlib.Path):
        features, target = self.vectorizer.load_data(data_dir)
        return Predict(target, self.estimator.predict(features))

    @staticmethod
    def load(data_dir: pathlib.Path):
        with open(data_dir / "model.pickle", "rb") as f:
            estimator = pickle.load(f)

        with open(data_dir / "vectorizer.pickle", "rb") as f:
            vect = pickle.load(f)

        return SkCls(vect, estimator)
