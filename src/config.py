from pathlib import Path

FORECAST_HORIZON = 48
SEASON = 24
DEFAULT_MODEL = 'SeasonalNaive'

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / '__output__'
INPUT_DIR = PROJECT_ROOT / '__input__' / 'Solar Power Plant'

PLANT1_GEN = INPUT_DIR / 'Plant_1_Generation_Data.csv'
PLANT1_WEATHER = INPUT_DIR / 'Plant_1_Weather_Sensor_Data.csv'
PLANT2_GEN = INPUT_DIR / 'Plant_2_Generation_Data.csv'
PLANT2_WEATHER = INPUT_DIR / 'Plant_2_Weather_Sensor_Data.csv'

TRAIN_PATH = OUTPUT_DIR / 'train.parquet'
TEST_PATH = OUTPUT_DIR / 'test.parquet'
HOURLY_PATH = OUTPUT_DIR / 'hourly.parquet'
