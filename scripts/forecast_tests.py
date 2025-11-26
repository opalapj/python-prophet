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


@process_and_write_to("yearly_auto.csv")
def add_auto_yearly_seasonality():
    global ts

    ts = ts.fcst.add_seasonality(
        kind="yearly",
        mode="auto",
    )


def main():
    read_limit_normalize()
    fit_predict_plot()
    add_auto_yearly_seasonality()


if __name__ == "__main__":
    main()
