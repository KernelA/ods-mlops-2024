import numpy as np
import polars as pl
from numpy import ndarray
from sklearn.base import TransformerMixin


class BooleanEncoder(TransformerMixin):
    def _to_numpy(self, x: pl.DataFrame | pl.Series):
        return x.to_numpy().astype(np.float32)

    def fit(self, X):
        return self

    def transform(self, X):
        return self._to_numpy(X)

    def get_params(self, **kwargs):
        return {}

    def fit_transform(self, X: pl.DataFrame | pl.Series, y=None, **fit_params) -> ndarray:
        return self._to_numpy(X)
