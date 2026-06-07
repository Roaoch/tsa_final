from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, Naive, SeasonalNaive

from .config import DEFAULT_MODEL, FORECAST_HORIZON, OUTPUT_DIR, SEASON
from .data import load_train_test, to_forecast_frame
from .metrics import evaluate


@dataclass
class PipelineResult:
    forecasts: pd.DataFrame
    metrics: dict[str, float]
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
    sf: StatsForecast | None = field(default=None, init=False)

    def _build_models(self) -> list:
        builders = {
            'AutoTheta': lambda: AutoTheta(season_length=self.season_length),
            'AutoETS': lambda: AutoETS(season_length=self.season_length),
            'AutoARIMA': lambda: AutoARIMA(season_length=self.season_length),
            'SeasonalNaive': lambda: SeasonalNaive(season_length=self.season_length),
            'Naive': lambda: Naive(),
        }
        if self.model_name not in builders:
            raise ValueError(
                f'Unknown model {self.model_name!r}. '
                f'Choose from: {sorted(builders)}'
            )
        return [builders[self.model_name]()]

    def fit(self, train: pd.DataFrame) -> SolarForecastPipeline:
        self.sf = StatsForecast(
            models=self._build_models(),
            freq='h',
            n_jobs=-1,
        )
        self.sf.fit(train[['unique_id', 'ds', 'y']])
        return self

    def predict(self) -> pd.DataFrame:
        if self.sf is None:
            raise RuntimeError('Pipeline is not fitted.')
        t0 = time.perf_counter()
        fcst = self.sf.predict(h=self.horizon)
        self._last_predict_seconds = time.perf_counter() - t0
        return fcst

    def run(
        self,
        train_df: pd.DataFrame | None = None,
        test_df: pd.DataFrame | None = None,
    ) -> PipelineResult:
        if train_df is None or test_df is None:
            train_df, test_df = load_train_test()

        train = to_forecast_frame(train_df)
        test = to_forecast_frame(test_df)

        t0 = time.perf_counter()
        self.fit(train)
        fit_seconds = time.perf_counter() - t0

        forecasts = self.predict()
        model_col = self.model_name
        if model_col not in forecasts.columns:
            model_col = forecasts.columns.difference(['unique_id', 'ds']).tolist()[0]

        merged = test.merge(forecasts, on=['unique_id', 'ds'], how='left')
        eval_df = merged.dropna(subset=[model_col])
        metrics = evaluate(eval_df['y'].values, eval_df[model_col].values)

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        forecasts.to_parquet(self.artifact_dir / 'latest_forecast.parquet', index=False)
        merged.to_parquet(self.artifact_dir / 'latest_evaluation.parquet', index=False)
        with open(self.artifact_dir / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'model': self.model_name,
                    'horizon': self.horizon,
                    'metrics': metrics,
                    'fit_seconds': fit_seconds,
                    'predict_seconds': getattr(self, '_last_predict_seconds', np.nan),
                },
                f,
                indent=2,
            )

        return PipelineResult(
            forecasts=forecasts,
            metrics=metrics,
            fit_seconds=fit_seconds,
            predict_seconds=getattr(self, '_last_predict_seconds', 0.0),
            model_name=self.model_name,
        )
