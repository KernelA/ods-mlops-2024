from dataclasses import dataclass

from omegaconf import MISSING

BOOL_COLUMNS = frozenset(
    [
        "root_stone",
        "root_grate",
        "root_other",
        "trunk_wire",
        "trnk_light",
        "trnk_other",
        "brch_light",
        "brch_shoe",
        "brch_other",
    ]
)


@dataclass
class DataConfig:
    input_path: str
    out_path: str


@dataclass
class ConvertConfig:
    data_convert: DataConfig = MISSING
