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
