import calendar

import pandas as pd

from forecast.classes import Period


# For ForecastAccessor.
INDEX_NAME = "ds"
EXOGENOUS_VARIABLE_NAME = "y"

# For ConditionAccessor.
INDEX_TYPE = pd.DatetimeIndex
SEASON = Period(
    name="season",
    mapping={
        1: "summer",
        2: "fall",
        3: "winter",
        4: "spring",
    },
)
MONTH = Period(
    name="month",
    mapping={m.value: m.name.lower() for m in calendar.Month},
)
DAYTYPE = Period(
    name="daytype",
    mapping={
        1: "workday",
        2: "weekend",
    },
)
WEEKDAY = Period(
    name="weekday",
    mapping={d.value + 1: d.name.lower() for d in calendar.Day},
)
HOUR = Period(
    name="hour",
    mapping={h: str(h) for h in range(1, 25)},
)
