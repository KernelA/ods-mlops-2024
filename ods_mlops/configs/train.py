from dataclasses import dataclass
from typing import Any


@dataclass
class TrainConfig:
    input_dir: str
    out_dir: str
    model: Any
