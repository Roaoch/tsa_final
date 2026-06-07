import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import STL


def iqr_anomaly_scores(series: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (series < lower) | (series > upper)


def stl_zscore_anomalies(
    series: pd.Series,
    period: int = 24,
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    stl = STL(series, period=period, robust=True).fit()
    remainder = stl.resid
    z = (remainder - remainder.mean()) / remainder.std(ddof=0)
    result = pd.DataFrame(
        {
            'observed': series,
            'trend': stl.trend,
            'seasonal': stl.seasonal,
            'remainder': remainder,
            'z_score': z,
            'is_anomaly': np.abs(z) > z_threshold,
        },
        index=series.index,
    )
    return result


def isolation_forest_anomalies(
    features: pd.DataFrame,
    contamination: float = 0.02,
    random_state: int = 42,
) -> pd.DataFrame:
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
    )
    preds = model.fit_predict(features.fillna(features.median()))
    scores = model.decision_function(features.fillna(features.median()))
    return pd.DataFrame(
        {
            'anomaly_score': scores,
            'is_anomaly': preds == -1,
        },
        index=features.index,
    )
