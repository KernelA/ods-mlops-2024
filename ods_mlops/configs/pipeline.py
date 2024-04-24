from dataclasses import dataclass, field
from typing import Any, List

from omegaconf import MISSING

from .data_config import DataConfig
from .preprocessing import DataInfoConfig

DEFAULTS = [{"data_convert": "parquet", "preprocessing": "base"}]


@dataclass
class PipelineConfig:
    defaults: List[Any] = field(default_factory=lambda: DEFAULTS)
    data_convert: DataConfig = MISSING
    preprocessing: DataInfoConfig = MISSING
