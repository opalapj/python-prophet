from dataclasses import dataclass
from dataclasses import field

import pandas as pd
import prophet as ph


@dataclass
class Period:
    name: str
    mapping: dict


@dataclass
class Seasonality:
    kind: str
    mode: str
    period: float
    fourier_order: float
    conditions: tuple


@dataclass
class Shock:
    description: str
    frame: pd.DataFrame


@dataclass
class Regressor:
    description: str
    span: pd.DatetimeIndex
    conditions: tuple


@dataclass
class Model:
    seasonalities: list = field(default_factory=list)
    regressors: list = field(default_factory=list)
    shocks: list = field(default_factory=list)
    country: str = None
    fit: ph.Prophet = None
    forecast: pd.DataFrame = None
