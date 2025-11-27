import pandas as pd

import forecast

from setup import locations
from setup import settings


ts: pd.DataFrame


def read_limit_normalize():
    global ts

    ts = forecast.read_time_series(
        input_filepath=locations.input / settings.training,
    )
    ts = ts.fcst.limit_training_set(
        start_date=settings.training_start_date,
        end_date=settings.training_end_date,
    )
    ts = ts.fcst.normalize_index()


def fit_predict_plot():
    global ts

    ts = ts.fcst.fit_model()
    ts = ts.fcst.predict(
        number_of_forecast_years=settings.number_of_forecast_years,
        first_day_of_forecast=settings.forecast_start_date,
        include_training_years=settings.include_training_years,
    )
    ts.fcst.plot()


def process_and_write_to(csv):
    def decorator(fun):
        def wrapper():
            read_limit_normalize()
            fun()
            fit_predict_plot()
            ts.fcst.write_time_series(
                output_filepath=locations.output / csv,
            )

        return wrapper

    return decorator


@process_and_write_to("1_yearly_auto.csv")
def add_yearly_auto_seasonality():
    global ts

    ts = ts.fcst.add_seasonality(
        kind="yearly",
        mode="auto",
    )


@process_and_write_to("2_yearly_auto_weekly_auto.csv")
def add_yearly_auto_weekly_auto_seasonality():
    global ts

    ts = ts.fcst.add_seasonality(
        kind="yearly",
        mode="auto",
    )
    ts = ts.fcst.add_seasonality(
        kind="weekly",
        mode="auto",
    )


@process_and_write_to("3_yearly_auto_weekly_auto_daily_auto.csv")
def add_yearly_auto_weekly_auto_daily_auto_seasonality():
    global ts

    ts = ts.fcst.add_seasonality(
        kind="yearly",
        mode="auto",
    )
    ts = ts.fcst.add_seasonality(
        kind="weekly",
        mode="auto",
    )
    ts = ts.fcst.add_seasonality(
        kind="daily",
        mode="auto",
    )


@process_and_write_to("4_full_conditional_seasonalities_one_by_one.csv")
def add_full_conditional_seasonalities_one_by_one():
    global ts

    ts = ts.fcst.add_seasonality(
        kind="yearly",
        mode="force",
        conditions=None,
    )
    ts = ts.fcst.add_seasonality(
        kind="weekly",
        mode="force",
        conditions=("month",),
        # conditions=("season",),
    )
    ts = ts.fcst.add_seasonality(
        kind="daily",
        mode="force",
        conditions=("month", "weekday"),
        # conditions=("month", "daytype"),
        # conditions=("season", "weekday"),
        # conditions=("season", "daytype"),
    )


@process_and_write_to("5_full_conditional_seasonalities_at_once.csv")
def add_full_conditional_seasonalities_at_once():
    global ts

    ts = ts.fcst.add_seasonalities(
        {
            "kind": "yearly",
            "mode": "force",
            "conditions": None,
        },
        {
            "kind": "weekly",
            "mode": "force",
            "conditions": ("month",),
        },
        {
            "kind": "daily",
            "mode": "force",
            "conditions": ("month", "weekday"),
        },
    )


@process_and_write_to("6_add_shocks.csv")
def add_shocks():
    global ts

    ts = ts.fcst.add_seasonalities(
        {
            "kind": "yearly",
            "mode": "force",
            "conditions": None,
        },
        {
            "kind": "weekly",
            "mode": "force",
            "conditions": ("month",),
        },
        {
            "kind": "daily",
            "mode": "force",
            "conditions": ("month", "weekday"),
        },
    )
    ts = ts.fcst.add_shocks(
        {"description": "dec_23", "spans": (("2023-12-18", "2023-12-31"),)},
    )


@process_and_write_to("7_regressors.csv")
def add_regressors():
    global ts

    ts = ts.fcst.add_seasonalities(
        {
            "kind": "yearly",
            "mode": "force",
            "conditions": None,
        },
        {
            "kind": "weekly",
            "mode": "force",
            "conditions": ("month",),
        },
        {
            "kind": "daily",
            "mode": "force",
            "conditions": ("month", "weekday"),
        },
    )
    ts = ts.fcst.add_shocks(
        {"description": "dec_23", "spans": (("2023-12-18", "2023-12-31"),)},
    )
    ts = ts.fcst.add_regressors(
        {
            "description": "wdb",
            "spans": (
                ("2024-06-24", "2024-08-04"),
                ("2024-08-05", "2024-09-01"),
            ),
        },
    )


@process_and_write_to("8_country_holidays.csv")
def add_country_holidays():
    global ts

    ts = ts.fcst.add_seasonalities(
        {
            "kind": "yearly",
            "mode": "force",
            "conditions": None,
        },
        {
            "kind": "weekly",
            "mode": "force",
            "conditions": ("month",),
        },
        {
            "kind": "daily",
            "mode": "force",
            "conditions": ("month", "weekday"),
        },
    )
    ts = ts.fcst.add_shocks(
        {"description": "dec_23", "spans": (("2023-12-18", "2023-12-31"),)},
    )
    ts = ts.fcst.add_regressors(
        {
            "description": "wdb",
            "spans": (
                ("2024-06-24", "2024-08-04"),
                ("2024-08-05", "2024-09-01"),
            ),
        },
    )
    ts = ts.fcst.add_country_holidays(settings.country)


@process_and_write_to("9_match_timezone.csv")
def match_timezone():
    global ts
    ts = forecast.read_time_series(
        input_filepath=locations.input / settings.training,
    )
    ts = ts.fcst.limit_training_set(
        start_date=settings.training_start_date,
        end_date=settings.training_end_date,
    )
    ts = ts.fcst.normalize_index()
    ts = ts.fcst.fit_model()
    ts = ts.fcst.predict(
        number_of_forecast_years=settings.number_of_forecast_years,
        first_day_of_forecast=settings.forecast_start_date,
        include_training_years=settings.include_training_years,
    )
    ts = ts.fcst.match_tz(settings.tz)


def main():
    add_yearly_auto_seasonality()
    add_yearly_auto_weekly_auto_seasonality()
    add_yearly_auto_weekly_auto_daily_auto_seasonality()
    add_full_conditional_seasonalities_one_by_one()
    add_full_conditional_seasonalities_at_once()
    add_shocks()
    add_regressors()
    add_country_holidays()
    match_timezone()


if __name__ == "__main__":
    main()
