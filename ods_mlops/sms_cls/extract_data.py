from typing import Iterable
from unicodedata import normalize

import polars as pl


def extract_data(file_object: Iterable[str], class_mapping: dict) -> pl.DataFrame:
    messages = []

    for line in map(str.strip, file_object):
        if not line:
            continue

        index = -1
        for i, c in enumerate(line):
            if c == "\t":
                index = i

        if index == -1:
            continue

        text, label = line[:index], line[index + 1 :]
        messages.append((int(label.strip().lower()), normalize("NFKD", text.strip())))

    data = pl.from_records(messages, schema={"target": pl.Int8, "text": pl.String})
    return data


def extract_data_df(file_object: Iterable[str], class_mapping: dict) -> pl.DataFrame:
    data = (
        pl.from_records(list(file_object), schema={"text": pl.Utf8})
        .lazy()
        .with_columns(pl.col("text").str.strip_chars())
        .filter(pl.col("text").str.len_chars() > 0)
        .with_columns(pl.col("text").str.extract_groups(r"(?P<text>.+)\t(?P<target>\d+)"))
        .collect()
    )

    if not data.is_empty():
        return (
            data.unnest("text")
            .select("target", "text")
            .with_columns(pl.col("target").cast(pl.Int8))
            .with_columns(
                pl.col("text").map_elements(
                    lambda x: normalize("NFKD", x.strip()), return_dtype=pl.Utf8
                )
            )
        )

    return pl.DataFrame(schema={"target": pl.Int8, "text": pl.Utf8})
