from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, NBEATS, NHITS
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, Naive, SeasonalNaive

from .config import DEFAULT_MODEL, FORECAST_HORIZON, OUTPUT_DIR, SEASON
from .data import load_train_test, to_forecast_frame, to_ml_frame
from .metrics import evaluate

STATS_MODELS = frozenset({'Naive', 'SeasonalNaive', 'AutoTheta', 'AutoETS', 'AutoARIMA'})
ML_MODELS = frozenset({'LGBMRegressor', 'RandomForestRegressor', 'Ridge'})
DL_MODELS = frozenset({'NHITS', 'NBEATS', 'LSTM'})
SUPPORTED_MODELS = STATS_MODELS | ML_MODELS | DL_MODELS


@dataclass
class PipelineResult:
    forecasts: pd.DataFrame
    metrics: dict[str, float]
    metrics_no_zeros: dict[str, float]
    fit_seconds: float
    predict_seconds: float
    model_name: str = DEFAULT_MODEL


@dataclass
class SolarForecastPipeline:
    """Offline batch pipeline for plant-level hourly AC power forecasting."""

    horizon: int = FORECAST_HORIZON
    season_length: int = SEASON
    model_name: str = DEFAULT_MODEL
    artifact_dir: Path = field(default_factory=lambda: OUTPUT_DIR / 'pipeline')
    sf: Any | None = field(default=None, init=False)

    def _build_stats_model(self):
        builders = {
            'AutoTheta': lambda: AutoTheta(season_length=self.season_length),
            'AutoETS': lambda: AutoETS(season_length=self.season_length),
            'AutoARIMA': lambda: AutoARIMA(season_length=self.season_length),
            'SeasonalNaive': lambda: SeasonalNaive(season_length=self.season_length),
            'Naive': lambda: Naive(),
        }
        return builders[self.model_name]()

    def _build_ml_model(self):
        builders = {
            'LGBMRegressor': lambda: LGBMRegressor(
                n_estimators=300, learning_rate=0.05, random_state=42,
            ),
            'RandomForestRegressor': lambda: RandomForestRegressor(
                n_estimators=200, random_state=42, n_jobs=-1,
            ),
            'Ridge': lambda: Ridge(alpha=1.0),
        }
        return builders[self.model_name]()

    def _build_dl_model(self):
        builders = {
            'NHITS': lambda: NHITS(h=self.horizon, input_size=72, max_steps=200),
            'NBEATS': lambda: NBEATS(h=self.horizon, input_size=72, max_steps=200),
            'LSTM': lambda: LSTM(h=self.horizon, input_size=72, max_steps=200),
        }
        return builders[self.model_name]()

    def _build_mlf(self) -> MLForecast:
        return MLForecast(
            models=[self._build_ml_model()],
            freq='h',
            lags=[1, 2, 6, 12, 24],
            lag_transforms={
                1: [ExpandingMean()],
                24: [RollingMean(window_size=3)],
            },
            date_features=['hour', 'dayofweek'],
        )

    def fit(self, train_df: pd.DataFrame) -> SolarForecastPipeline:
        if self.model_name in DL_MODELS:
            self.sf = NeuralForecast(models=[self._build_dl_model()], freq='h')
            self.sf.fit(to_ml_frame(train_df))
        elif self.model_name in ML_MODELS:
            self.sf = self._build_mlf()
            self.sf.fit(to_ml_frame(train_df), static_features=[])
        else:
            self.sf = StatsForecast(
                models=[self._build_stats_model()],
                freq='h',
                n_jobs=-1,
            )
            self.sf.fit(to_forecast_frame(train_df))
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        if self.sf is None:
            raise RuntimeError('Pipeline is not fitted.')

        t0 = time.perf_counter()
        if self.model_name in DL_MODELS:
            fcst = self.sf.predict(h=self.horizon, futr_df=to_ml_frame(test_df))
        elif self.model_name in ML_MODELS:
            fcst = self.sf.predict(h=self.horizon, X_df=to_ml_frame(test_df))
        else:
            fcst = self.sf.predict(h=self.horizon)
        self._last_predict_seconds = time.perf_counter() - t0
        return fcst

    def run(
        self,
        train_df: pd.DataFrame | None = None,
        test_df: pd.DataFrame | None = None,
    ) -> PipelineResult:
        if self.model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f'Unknown model {self.model_name!r}. '
                f'Choose from: {sorted(SUPPORTED_MODELS)}'
            )

        if train_df is None or test_df is None:
            train_df, test_df = load_train_test()

        test = to_forecast_frame(test_df)

        t0 = time.perf_counter()
        self.fit(train_df)
        fit_seconds = time.perf_counter() - t0

        forecasts = self.predict(test_df)
        model_col = self.model_name
        if model_col not in forecasts.columns:
            model_col = forecasts.columns.difference(['unique_id', 'ds']).tolist()[0]

        merged = test.merge(forecasts, on=['unique_id', 'ds'], how='left')
        eval_df = merged.dropna(subset=[model_col])
        metrics = evaluate(eval_df['y'].values, eval_df[model_col].values)
        metrics_no_zeros = evaluate(
            eval_df['y'].values, eval_df[model_col].values, no_zeros=True,
        )

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        forecasts.to_parquet(self.artifact_dir / 'latest_forecast.parquet', index=False)
        merged.to_parquet(self.artifact_dir / 'latest_evaluation.parquet', index=False)
        with open(self.artifact_dir / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'model': self.model_name,
                    'horizon': self.horizon,
                    'metrics': metrics,
                    'metrics_no_zeros': metrics_no_zeros,
                    'fit_seconds': fit_seconds,
                    'predict_seconds': getattr(self, '_last_predict_seconds', np.nan),
                },
                f,
                indent=2,
            )

        return PipelineResult(
            forecasts=forecasts,
            metrics=metrics,
            metrics_no_zeros=metrics_no_zeros,
            fit_seconds=fit_seconds,
            predict_seconds=getattr(self, '_last_predict_seconds', 0.0),
            model_name=self.model_name,
        )
