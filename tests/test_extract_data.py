from datetime import timedelta

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from ods_mlops.sms_cls.extract_data import extract_data


@settings(deadline=timedelta(seconds=3))
@given(
    pairs=st.lists(
        st.tuples(
            st.sampled_from([0, 1]),
            st.text(st.characters(exclude_categories=["Zl", "Cf"]), min_size=1, max_size=100),
        ),
        min_size=0,
        max_size=200,
    )
)
def test_extract_data(pairs):
    lines = list(map(lambda x: f"a{x[1]}\t{x[0]}", pairs))
    class_mapping = {"a": 1}
    df = extract_data(lines, class_mapping)

    assert isinstance(df, pl.DataFrame)
    assert df.columns == ["target", "text"]

    # Test data types in columns
    assert df["target"].dtype == pl.Int8
    assert df["text"].dtype == pl.String

    # Test number of rows
    assert len(df) == len(lines)

    if pairs:
        assert df["target"].to_list() == list(tuple(zip(*pairs))[0])
