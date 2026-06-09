
import pandas as pd

from .config import (
    FORECAST_HORIZON,
    HOURLY_PATH,
    PLANT1_GEN,
    PLANT1_WEATHER,
    PLANT2_GEN,
    PLANT2_WEATHER,
    OUTPUT_DIR,
    TEST_PATH,
    TRAIN_PATH,
    WEATHER_COLS,
)


def prepare_generation(
    path: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    day_first=False
):
    df = pd.read_csv(path)
    df = df.drop([
        'DAILY_YIELD',
        'TOTAL_YIELD',
        'SOURCE_KEY',
        'PLANT_ID',
        'DC_POWER'
    ], axis=1)

    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], format='mixed', dayfirst=day_first)

    df = df.loc[
        df['DATE_TIME'].between(
            start,
            end
        )
    ]
    df = df.groupby(['DATE_TIME']).sum().reset_index() #type: ignore
    df = df.set_index('DATE_TIME')

    df: pd.DataFrame = df.resample('1h').sum()

    return df


def prepare_wether(path: str):
    df = pd.read_csv(path)
    df = df.drop([
        'SOURCE_KEY',
        'PLANT_ID'
    ], axis=1)

    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], format='mixed')
    df = df.set_index('DATE_TIME')

    df: pd.DataFrame = df.resample('1h').mean()

    return df


def prepare_datasets(gen_path: str, wether_path: str, day_first=False):
    df_wether = prepare_wether(wether_path)
    df_gen = prepare_generation(
        gen_path,
        df_wether.index.min(), #type: ignore
        df_wether.index.max(), #type: ignore
        day_first
    )

    df = df_gen.merge(df_wether, how='left', left_index=True, right_index=True)
    df = df.interpolate('time').ffill().bfill()

    return df


def train_test_split(df: pd.DataFrame, holdout: int):
    test_times = df.index.unique().sort_values()[-holdout:] #type: ignore
    test = df.loc[df.index.isin(test_times)]
    train = df.loc[~df.index.isin(test_times)]
    return train, test


def build_datasets(holdout: int = FORECAST_HORIZON, save: bool = True):
    """Load Plant 1+2, merge, split and optionally save parquet files."""
    df1 = prepare_datasets(str(PLANT1_GEN), str(PLANT1_WEATHER), day_first=True)
    df2 = prepare_datasets(str(PLANT2_GEN), str(PLANT2_WEATHER))
    df1['unique_id'] = '1'
    df2['unique_id'] = '2'
    df = pd.concat([df1, df2])

    train, test = train_test_split(df, holdout)
    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        train.to_parquet(TRAIN_PATH)
        test.to_parquet(TEST_PATH)
        df.to_parquet(HOURLY_PATH)

    return train, test


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        return build_datasets(save=True)
    return pd.read_parquet(TRAIN_PATH), pd.read_parquet(TEST_PATH)


def to_forecast_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        'unique_id': df['unique_id'].astype(str),
        'ds': df.index,
        'y': df['AC_POWER'],
    })


def to_ml_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Long format for mlforecast / neuralforecast with weather exogenous."""
    frame = to_forecast_frame(df)
    for col in WEATHER_COLS:
        frame[col] = df[col].values
    return frame
