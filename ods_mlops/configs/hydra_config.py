from hydra.core.config_store import ConfigStore

from .data_config import DataConfig
from .pipeline import PipelineConfig

cs = ConfigStore.instance()
cs.store(name="pipeline", node=PipelineConfig)
cs.store(name="parquet", group="data_convert", node=DataConfig)
