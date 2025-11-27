import forecast

from setup import locations
from setup import settings


def main():
    ts = forecast.read_time_series(
        input_filepath=locations.input / settings.training,
    )
    ts = ts.fcst.limit_training_set(
        start_date=settings.training_start_date,
        end_date=settings.training_end_date,
    )
    ts = ts.fcst.normalize_index()
    ts.index.cond.get_condition("season")
    ts.index.cond.get_condition("season", dummy=False)
    ts.index.cond.get_condition("month")
    ts.index.cond.get_condition("month", dummy=False)
    ts.index.cond.get_condition("daytype")
    ts.index.cond.get_condition("daytype", dummy=False)
    ts.index.cond.get_condition("weekday")
    ts.index.cond.get_condition("weekday", dummy=False)
    ts.index.cond.get_condition("hour")
    ts.index.cond.get_condition("hour", dummy=False)
    ts.index.cond.get_conditions(["season", "daytype"])
    ts.index.cond.get_conditions(["season", "daytype"], combined=False)
    ts.index.cond.get_conditions(["season", "daytype"], dummy=False)
    ts.index.cond.get_conditions(["season", "daytype"], dummy=False, combined=False)


if __name__ == "__main__":
    main()
