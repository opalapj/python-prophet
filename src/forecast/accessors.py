import numpy as np
import pandas as pd
import prophet as ph
from matplotlib import pyplot as plt

from forecast.classes import Regressor
from forecast.classes import Seasonality
from forecast.classes import Shock
from forecast.constants import DAYTYPE
from forecast.constants import EXOGENOUS_VARIABLE_NAME
from forecast.constants import HOUR
from forecast.constants import INDEX_NAME
from forecast.constants import INDEX_TYPE
from forecast.constants import MONTH
from forecast.constants import SEASON
from forecast.constants import WEEKDAY
from forecast.decorators import _work_on_copy
from forecast.exceptions import ConditionKindError
from forecast.exceptions import SeasonalityKindError
from forecast.exceptions import SeasonalityModeError
from forecast.helpers import match_tz
from forecast.helpers import merge_spans


@pd.api.extensions.register_dataframe_accessor("fcst")
class ForecastAccessor:

    def __init__(self, obj):
        self._validate(obj)
        self._obj = obj
        self.model = None

    @staticmethod
    def _validate(obj):
        if (obj.index.name != INDEX_NAME) or (
            EXOGENOUS_VARIABLE_NAME not in obj.columns
        ):
            raise ValueError(
                f"Time series must have {INDEX_NAME!r} as index name and "
                f"{EXOGENOUS_VARIABLE_NAME!r} as column name with exogenous "
                f"variable values."
            )

    @_work_on_copy
    def limit_training_set(self, start_date, end_date):
        df = self._obj
        df = df.loc[start_date:end_date]
        return df

    @_work_on_copy
    def normalize_index(self):
        # return: dataframe
        df = self._obj
        # Insert an empty index for the missing hour at the change to
        # daylight saving time and average the values from repeated hours
        # at the change from daylight saving time.
        df = df.resample("h").mean()
        # Filling in the value for the missing hour.
        df = df.interpolate()
        return df

    @_work_on_copy
    def add_seasonality(self, kind, mode, conditions=None):
        # `kind`: str, one of: 'yearly', 'weekly', 'daily'
        # `mode`: str, one of: 'auto', 'force'; if None -> 'auto'
        # `conditions`: tuple, combination of 'season', 'month', 'daytype', 'weekday'; e.g. ('month',), ('season', 'daytype')
        # return: dataframe
        if kind not in ("yearly", "weekly", "daily"):
            raise SeasonalityKindError(
                f"There is no available seasonality kind like {kind!r}."
            )
        if mode not in ("auto", "force"):
            raise SeasonalityModeError(
                f"There is no available seasonality mode like {mode!r}."
            )
        match mode:
            case "auto":
                period = None
                fourier_order = None
                conditions = None
            case "force":
                match kind:
                    case "yearly":
                        period = 365.25
                        fourier_order = 10
                        conditions = None  # No possible conditions.
                    case "weekly":
                        period = 7
                        fourier_order = 3
                    case "daily":
                        period = 1
                        fourier_order = 4
        self.model.seasonalities.append(
            Seasonality(
                kind=kind,
                mode=mode,
                period=period,
                fourier_order=fourier_order,
                conditions=conditions,
            )
        )
        return self._obj

    def add_yearly_seasonality(self, **kwargs):
        # `kwargs`: kwargs for `add_seasonality` method except for the kind arg
        return self.add_seasonality(kind="yearly", **kwargs)

    def add_weekly_seasonality(self, **kwargs):
        # `kwargs`: kwargs for `add_seasonality` method except for the kind arg
        return self.add_seasonality(kind="weekly", **kwargs)

    def add_daily_seasonality(self, **kwargs):
        # `kwargs`: kwargs for `add_seasonality` method except for the kind arg
        return self.add_seasonality(kind="daily", **kwargs)

    def add_seasonalities(self, *seasonalities):
        # `seasonalities`: dicts with kwargs for `add_seasonality` method
        df = self._obj
        for seasonality in seasonalities:
            df = df.fcst.add_seasonality(**seasonality)
        return df

    @_work_on_copy
    def add_shock(self, description, spans):
        # `description`: str
        # `spans`: tuple of tuples
        # return: dataframe
        model = self.model
        data = []
        for span in spans:
            span = [pd.Timestamp(ds) for ds in span]
            data.append(
                {
                    "holiday": description,
                    "ds": span[0],
                    "ds_end": span[1],
                }
            )
        frame = pd.DataFrame(data=data)
        frame["span"] = frame["ds_end"] - frame["ds"]
        frame["upper_window"] = frame["span"].dt.days
        frame["lower_window"] = 0
        model.shocks.append(
            Shock(
                description=description,
                frame=frame,
            )
        )
        return self._obj

    def add_shocks(self, *shocks):
        # `shocks`: dicts with kwargs for `add_shock` method
        # return: dataframe
        df = self._obj
        for shock in shocks:
            df = df.fcst.add_shock(**shock)
        return df

    @_work_on_copy
    def add_regressor(self, description, spans):
        # `description`: str
        # `spans`: tuple of tuples
        # return: dataframe
        model = self.model
        conds_to_test = [
            ("month", "weekday"),
            ("month", "daytype"),
            ("season", "weekday"),
            ("season", "daytype"),
            ("weekday",),
            ("daytype",),
        ]
        span = merge_spans(*spans)
        conds = ("hour",)
        in_training = span.isin(self._obj.index)
        if in_training.all():
            conds = (*conds_to_test[0], *conds)
        else:
            for test_conds in conds_to_test:
                in_conds = span[in_training].cond.get_unique_conditions(
                    kinds=test_conds,
                )
                out_conds = span[~in_training].cond.get_unique_conditions(
                    kinds=test_conds,
                )
                if np.isin(out_conds, in_conds).all():
                    conds = (*test_conds, *conds)
                    break
        model.regressors.append(
            Regressor(
                description=description,
                span=span,
                conditions=conds,
            )
        )
        return self._obj

    def add_regressors(self, *regressors):
        # `regressors`: dicts with kwargs for `add_regressor` method
        # return: dataframe
        df = self._obj
        for regressor in regressors:
            df = df.fcst.add_regressor(**regressor)
        return df

    @_work_on_copy
    def add_country_holidays(self, country):
        # `country`: str, country code
        # return: dataframe
        model = self.model
        model.country = country
        return self._obj

    @_work_on_copy
    def fit_model(self):
        # return: dataframe
        df = self._obj
        model = self.model
        init_kwargs = {
            "growth": "flat",
            "yearly_seasonality": False,
            "weekly_seasonality": False,
            "daily_seasonality": False,
            "scaling": "minmax",
        }
        # Handling seasonalities with: (1) 'auto' mode and (2) 'force' mode
        # without conditions.
        for seasonality in model.seasonalities:
            if seasonality.mode == "auto":
                kwarg = "_".join((seasonality.kind, "seasonality"))
                init_kwargs[kwarg] = "auto"
            elif (seasonality.mode == "force") and (seasonality.conditions is None):
                kwarg = "_".join((seasonality.kind, "seasonality"))
                init_kwargs[kwarg] = True
        # Handling shocks.
        if model.shocks:
            frames = [shock.frame for shock in model.shocks]
            init_kwargs["holidays"] = pd.concat(frames)
        # Model initialization.
        model_ = ph.Prophet(**init_kwargs)
        # Handling holidays.
        if model.country:
            model_.add_country_holidays(country_name=model.country)
        # Handling seasonalities with 'force' mode with conditions.
        for seasonality in model.seasonalities:
            if (seasonality.mode == "force") and (seasonality.conditions is not None):
                conds_ = df.index.cond.get_conditions(seasonality.conditions)
                cond_names = conds_.columns
                for cond_name in cond_names:
                    model_.add_seasonality(
                        name="_".join((seasonality.kind, cond_name)),
                        period=seasonality.period,
                        fourier_order=seasonality.fourier_order,
                        condition_name=cond_name,
                    )
                if not np.isin(cond_names, df.columns).all():
                    df = df.join(conds_)
        # Handling regressors.
        for regressor in model.regressors:
            conds_ = regressor.span.cond.get_conditions(regressor.conditions)
            conds_.rename(
                columns=lambda name: regressor.description + "_" + name,
                inplace=True,
            )
            cond_names = conds_.columns
            for cond_name in cond_names:
                model_.add_regressor(cond_name)
            df = df.join(conds_.reindex(index=df.index, fill_value=False))
        # Model fitting.
        model.fit = model_.fit(df.reset_index())
        return df

    @_work_on_copy
    def predict(
        self, number_of_forecast_years, first_day_of_forecast, include_training_years
    ):
        # `number_of_forecast_years`: int
        # `first_day_of_forecast`: date str in ISO 8601 format
        # `include_training_years`: boolean
        # return: dataframe
        df = self._obj
        model = self.model
        first_day_of_forecast = pd.Timestamp(first_day_of_forecast)
        if include_training_years:
            start = df.index.min()
        else:
            start = first_day_of_forecast
        index = pd.date_range(
            start=start,
            end=first_day_of_forecast
            + pd.offsets.YearEnd(number_of_forecast_years)
            + pd.DateOffset(days=1),
            freq="h",
            name="ds",
            inclusive="left",
        )
        future = pd.DataFrame(index=index)
        # Handling seasonalities with 'force' mode with conditions.
        for seasonality in model.seasonalities:
            if (seasonality.mode == "force") and (seasonality.conditions is not None):
                conds_ = index.cond.get_conditions(seasonality.conditions)
                cond_names = conds_.columns
                if not np.isin(cond_names, future.columns).all():
                    future = future.join(conds_)
        # Handling regressors.
        for regressor in model.regressors:
            conds_ = regressor.span.cond.get_conditions(regressor.conditions)
            conds_.rename(
                columns=lambda name: regressor.description + "_" + name,
                inplace=True,
            )
            future = future.join(conds_.reindex(index=index, fill_value=False))
        model._forecast = model.fit.predict(future.reset_index())
        model.forecast = model._forecast[["ds", "yhat"]].set_index("ds")
        return df

    @_work_on_copy
    def match_tz(self, tz):
        # `tz`: str, e.g. 'Europe/Warsaw'
        # return: dataframe
        df = self.model.forecast
        self.model.forecast = match_tz(df=df, tz=tz)
        return self._obj

    def plot(self):
        self.model.fit.plot(self.model._forecast)
        plt.show()

    def write_time_series(self, output_filepath):
        # `output_filepath`: str
        self.model.forecast.to_csv(path_or_buf=output_filepath)


@pd.api.extensions.register_index_accessor("cond")
class ConditionAccessor:

    def __init__(self, obj):
        self._validate(obj)
        self._obj = obj

    @staticmethod
    def _validate(obj):
        if not isinstance(obj, INDEX_TYPE):
            raise ValueError(f"Time series must have {INDEX_TYPE.__name__}-like index.")

    def get_condition(
        self,
        kind,
        mapping=None,
        name=None,
        dummy=True,
        start_month=None,
        weekend=None,
    ):
        # `kind`: str, one of: 'season', 'month', 'daytype', 'weekday', 'hour'
        # `mapping`: dict
        # `name`: str
        # `dummy`: bool
        # `start_month`: int
        # `weekend`: tuple of weekend days
        # return: dataframe
        match kind:
            case "season":
                start_month = start_month or 6
                data = (self._obj.month - start_month) % 12 // 3 + 1
                mapping = mapping or SEASON.mapping
                name = name or SEASON.name
            case "month":
                data = self._obj.month
                mapping = mapping or MONTH.mapping
                name = name or MONTH.name
            case "daytype":
                weekend = weekend or (6, 7)
                data = np.where((self._obj.dayofweek + 1).isin(weekend), 2, 1)
                mapping = mapping or DAYTYPE.mapping
                name = name or DAYTYPE.name
            case "weekday":
                data = self._obj.dayofweek + 1
                mapping = mapping or WEEKDAY.mapping
                name = name or WEEKDAY.name
            case "hour":
                data = self._obj.hour + 1
                mapping = mapping or HOUR.mapping
                name = name or HOUR.name
            case _:
                raise ConditionKindError(
                    f"There is no available condition kind like {kind!r}."
                )
        cond = pd.Series(data=data, index=self._obj, name=name)
        if mapping:
            cond = cond.map(mapping)
        cond = cond.to_frame()
        if dummy:
            cond = pd.get_dummies(cond, prefix=name)
        return cond

    def get_season(self, **kwargs):
        # `kwargs`: kwargs for `get_condition` method except for the kind arg
        # return: dataframe
        return self.get_condition(kind="season", **kwargs)

    def get_month(self, **kwargs):
        # `kwargs`: kwargs for `get_condition` method except for the kind arg
        # return: dataframe
        return self.get_condition(kind="month", **kwargs)

    def get_daytype(self, **kwargs):
        # `kwargs`: kwargs for `get_condition` method except for the kind arg
        # return: dataframe
        return self.get_condition(kind="daytype", **kwargs)

    def get_weekday(self, **kwargs):
        # `kwargs`: kwargs for `get_condition` method except for the kind arg
        # return: dataframe
        return self.get_condition(kind="weekday", **kwargs)

    def get_hour(self, **kwargs):
        # `kwargs`: kwargs for `get_condition` method except for the kind arg
        # return: dataframe
        return self.get_condition(kind="hour", **kwargs)

    def get_conditions(self, kinds, dummy=True, combined=True):
        # `kinds`: tuple, combination of `kind` argument for `get_condition` method; e.g. ('season', 'daytype')
        # `dummy`: bool
        # `combined`: bool
        # return: dataframe
        if len(kinds) == 1:
            (kind,) = kinds
            return self.get_condition(kind=kind, dummy=dummy)
        conds = []
        if not combined:
            for kind in kinds:
                cond = self.get_condition(kind, dummy=dummy)
                conds.append(cond)
            conds = pd.concat(conds, axis=1)
            return conds
        else:
            for kind in kinds:
                cond = self.get_condition(kind, dummy=False)
                conds.append(cond)
            conds = pd.concat(conds, axis=1)
            combined = conds.apply(lambda row: "_".join(row), axis=1)
            name = "_".join(kinds)
            combined = combined.to_frame(name=name)
            if dummy:
                combined = pd.get_dummies(combined, prefix=name)
            return combined

    def get_unique_conditions(self, kinds):
        # `kinds`: tuple, combination of `kind` argument for `get_condition` method; e.g. ('season', 'daytype')
        # return: ndarray
        unique_conds = (
            self.get_conditions(
                kinds=kinds,
                dummy=False,
            )
            .squeeze()
            .unique()
        )
        return unique_conds
