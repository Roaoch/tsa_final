import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from statsforecast import StatsForecast
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

from .metrics import evaluate, metrics_table


def cross_validate_models(
    df: pd.DataFrame,
    models: list,
    h: int,
    n_windows: int = 5,
    step_size: int = 24,
    freq: str = 'h',
    n_jobs: int = -1,
) -> pd.DataFrame:
    sf = StatsForecast(models=models, freq=freq, n_jobs=n_jobs)
    return sf.cross_validation(
        df=df,
        h=h,
        n_windows=n_windows,
        step_size=step_size,
    )


def cv_summary(cv_df: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    rows = []
    for name in model_names:
        metrics = evaluate(cv_df['y'].values, cv_df[name].values)
        metrics['model'] = name
        rows.append(metrics)
    return metrics_table(rows)


def cv_by_window(cv_df: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    rows = []
    for cutoff, group in cv_df.groupby('cutoff'):
        for name in model_names:
            metrics = evaluate(group['y'].values, group[name].values)
            rows.append({'cutoff': cutoff, 'model': name, **metrics})
    return pd.DataFrame(rows)


def in_sample_fitted(
    df: pd.DataFrame,
    models: list,
    freq: str = 'h',
    n_jobs: int = -1,
) -> pd.DataFrame:
    sf = StatsForecast(models=models, freq=freq, n_jobs=n_jobs)
    sf.forecast(df=df, h=1, fitted=True)
    return sf.forecast_fitted_values()


def with_residuals(fitted_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    out = fitted_df.copy()
    out['residual'] = out['y'] - out[model_name]
    out['model'] = model_name
    return out


def with_cv_residuals(cv_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    out = cv_df[['unique_id', 'ds', 'y', model_name]].copy()
    out['residual'] = out['y'] - out[model_name]
    out['model'] = model_name
    return out


def plot_residual_diagnostics(
    residual_df: pd.DataFrame,
    model_name: str,
    *,
    ds_col: str = 'ds',
    residual_col: str = 'residual',
):
    res = residual_df[residual_col].dropna()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(residual_df[ds_col], res, linewidth=0.8)
    axes[0, 0].axhline(0, color='k', linewidth=0.5)
    axes[0, 0].set_title(f'Остатки во времени — {model_name}')

    sns.histplot(res, kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Распределение остатков')

    plot_acf(res, lags=48, ax=axes[1, 0])
    axes[1, 0].set_title('ACF остатков')

    stats.probplot(res, dist='norm', plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q plot (нормальность)')

    plt.tight_layout()
    plt.show()

    lb = acorr_ljungbox(res, lags=[24, 48], return_df=True)
    print(f'Ljung–Box ({model_name}):')
    return lb


def plot_cv_rmse_by_window(
    cv_window_metrics: pd.DataFrame,
    model_names: list[str],
    *,
    title: str = 'RMSE по окнам cross-validation',
):
    fig, ax = plt.subplots(figsize=(10, 4))
    for name in model_names:
        subset = cv_window_metrics[cv_window_metrics['model'] == name]
        ax.plot(subset['cutoff'], subset['RMSE'], marker='o', label=name)
    ax.set_title(title)
    ax.set_xlabel('cutoff (конец train-окна)')
    ax.set_ylabel('RMSE')
    ax.legend(loc='best', fontsize=8)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
