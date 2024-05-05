import io
from unicodedata import normalize

import polars as pl


def extract_data(file_object: io.TextIOWrapper, class_mapping: dict) -> pl.DataFrame:
    messages = []

    for line in map(str.strip, file_object):
        if not line:
            continue
        label, text = line.split("\t", maxsplit=1)
        messages.append((label.strip().lower(), normalize(text, "NFKD")))

    data = pl.from_records(messages, schema={"target": pl.String, "text": pl.String})
    return data.select(pl.col("target").map_dict(class_mapping, return_dtype=pl.Int8))
