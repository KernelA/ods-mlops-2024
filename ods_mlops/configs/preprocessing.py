from dataclasses import dataclass, field
from typing import List

from omegaconf import MISSING

from .data_config import BOOL_COLUMNS


@dataclass
class DataInfoConfig:
    numeric_cols: List[str] = field(
        default_factory=lambda: ["x_sp", "y_sp", "tree_dbh", "stump_diam"]
    )
    target_col: str = "health"
    cat_cols: List[str] = field(
        default_factory=lambda: [
            "curb_loc",
            "steward",
            "sidewalk",
            "borocode",
            "zipcode",
        ]
    )
    bin_cols: List[str] = field(default_factory=lambda: list(BOOL_COLUMNS))


@dataclass
class SplitConfig:
    train_size: float = 0.8
    seed: int = MISSING


@dataclass
class PreprocessConfig:
    input_data: str = MISSING
    data_info: DataInfoConfig = MISSING
    split_info: SplitConfig = MISSING
    out_dir: str = MISSING
