import pathlib

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore

from ods_mlops.configs import BOOL_COLUMNS, ConvertConfig

cs = ConfigStore.instance()
cs.store(name="base_convert", node=ConvertConfig)


@hydra.main(config_path="configs", config_name="convert_data", version_base="1.3")
def main(config: ConvertConfig):
    out_path = pathlib.Path(config.data_convert.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = (
        pl.scan_csv(
            config.data_convert.input_path,
            encoding="utf8",
            dtypes={
                "created_at": pl.Datetime,
            },
            null_values="None",
        )
        .with_columns(
            pl.col(col).map_elements({"Yes": True, "No": False}.get, return_dtype=pl.Boolean)
            for col in BOOL_COLUMNS
        )
        .collect()
    )
    data.write_parquet(out_path, statistics=True)


if __name__ == "__main__":
    main()
