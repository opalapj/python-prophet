import pandas as pd

from forecast.constants import EXOGENOUS_VARIABLE_NAME
from forecast.constants import INDEX_NAME


def merge_spans(*spans):
    # `spans`: two-elements tuples consists start and end date str in ISO 8601 format
    # return: pd.DatetimeIndex
    data = []
    for span in spans:
        span = [pd.Timestamp(ds) for ds in span]
        data.append(
            pd.date_range(
                start=span[0],
                end=span[1] + pd.DateOffset(days=1),
                freq="h",
                inclusive="left",
            ).to_series()
        )
    span = pd.concat(data).index
    return span


def match_tz(df, tz):
    # `df`: dataframe with naive DatetimeIndex
    # `tz`: str, e.g. 'Europe/Warsaw'
    # return: dataframe with aware DatetimeIndex
    index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="h",
        tz=tz,
        name=df.index.name,
    ).tz_localize(tz=None)
    df = df.reindex(index=index)
    df = df.tz_localize(
        tz=tz,
        ambiguous="infer",
    )
    return df


def read_time_series(input_filepath):
    # `input_filepath`: str
    # return: dataframe
    df = pd.read_csv(
        filepath_or_buffer=input_filepath,
        header=0,
        names=[INDEX_NAME, EXOGENOUS_VARIABLE_NAME],
        index_col=INDEX_NAME,
        parse_dates=True,
    )
    return df
