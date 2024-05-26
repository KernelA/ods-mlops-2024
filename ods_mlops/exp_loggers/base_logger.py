from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union


class BaseExpLogger(ABC):
    def __init__(self, exp_name: str, run_name: Optional[str] = None):
        self.exp_name = exp_name
        self.run_name = run_name

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        pass

    @abstractmethod
    def log_metric(self, key: str, value: Union[int, float]):
        pass

    @abstractmethod
    def log_figure(self, figure, name):
        pass

    @abstractmethod
    def close(self):
        pass
