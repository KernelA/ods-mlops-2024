from typing import Any, Dict

import mlflow

from .base_logger import BaseExpLogger


class MLflowLogger(BaseExpLogger):
    def __init__(self, exp_name: str, tracking_url: str, run_name: str | None = None):
        super().__init__(exp_name, run_name)
        self.tracking_url = tracking_url
        self._active_run = None

    def init(self):
        mlflow.set_tracking_uri(uri=self.tracking_url)
        mlflow.set_experiment(self.exp_name)
        self._active_run = mlflow.start_run(self.run_name)

    def close(self):
        if self._active_run is not None:
            mlflow.end_run()

    def log_params(self, params: Dict[str, Any]):
        mlflow.log_params(params)

    def log_figure(self, figure, name):
        mlflow.log_figure(figure, name)

    def log_metric(self, key: str, value: int | float):
        mlflow.log_metric(key, value)
