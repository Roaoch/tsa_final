# Анализ временных рядов — Solar Power Plant

Прогноз суммарной AC-мощности солнечных блоков (Plant 1 и Plant 2) на **48 часов** (2 суток).

## Структура репозитория

| Файл / каталог | Содержание |
|----------------|------------|
| `0_preprocessing_eda.ipynb` | Подготовка данных, EDA, постановка задачи |
| `1_stats_models.ipynb` | Статистические модели (`statsforecast`), backtest, остатки |
| `2_dl_models.ipynb` | ML (`mlforecast`), DL (`neuralforecast`), аномалии, backtest, остатки |
| `3_pipeline.ipynb` | Пайплайн и тестирование производительности |
| `src/` | Переиспользуемый код (`data`, `metrics`, `anomaly`, `stats`, `pipeline`, `config`) |
| `__output__/` | Подготовленные parquet и артефакты пайплайна |
| `__input__/Solar Power Plant/` | Исходные CSV (Kaggle) |

## Описание временного ряда

- **Источник:** [Kaggle — Solar Power Plant](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data/data) (Plant 1 и Plant 2, Индия).
- **Период:** 15.05.2020 — 17.06.2020 (~34 суток).
- **Частота исходных данных:** 15 минут (инверторы + погодный датчик на блок).
- **Целевой ряд:** суммарная `AC_POWER` по инверторам блока, агрегированная до **1 часа** (kW).
- **Экзогенные признаки:** `IRRADIATION`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`.
- **Серии:** `unique_id` = `1` (Plant 1), `2` (Plant 2) — global-модель на двух рядах.

## Постановка задачи

| Параметр | Значение |
|----------|----------|
| Горизонт | 48 часов |
| Режим | Offline batch |
| Train / Test | все данные кроме последних 48 ч / последние 48 ч |
| Метрики | MAE, RMSE, MAPE, sMAPE |
| Сезонность | Суточная (period = 24) |

Суточная сезонность доминирует: ночью мощность ≈ 0, днём — пик по инсоляции. ADF-тест показывает нестационарность исходного ряда; для ML используется сезонная разность `Differences([24])`, для stats — сезонные компоненты.

## EDA (кратко)

- Пропуски после агрегации минимальны; часовой индекс регулярен.
- `seasonal_decompose` / STL выявляют суточный сезонный паттерн.
- ACF/PACF — значимые лаги 24, 48, …
- Сильная связь `AC_POWER` ↔ `IRRADIATION`.

Подробные графики — в `0_preprocessing_eda.ipynb`.

## Аномалии (`2_dl_models.ipynb`)

| Метод | Идея | Параметры | Plant 1 (train) |
|-------|------|-----------|-----------------|
| IQR | Выбросы по квантилям | k = 1.5 | 0 точек |
| STL + Z-score | Остатки после STL(period=24) | \|z\| > 3 | 24 точки |
| Isolation Forest | Многомерные выбросы (power + weather) | contamination = 0.02 | 16 точек |

**Выбор:** STL + Z-score — основной интерпретируемый метод (trend / seasonal / remainder). IQR на часовых данных почти не срабатывает из-за естественного диапазона мощности. Isolation Forest дополняет анализ многомерными отклонениями (например, аномальное сочетание температуры и выработки).

## Методы прогнозирования и результаты

### Baseline (stats)

| Модель | Hold-out RMSE | Комментарий |
|--------|---------------|-------------|
| Naive | ~39952 (CV) | Нижняя граница |
| SeasonalNaive | 10990 | Учитывает lag=24 |

### Статистические (`statsforecast`) — ≥5

| Модель | Режим | Hold-out RMSE | Backtest RMSE (h=24, 5 окон) |
|--------|-------|---------------|------------------------------|
| AutoETS | auto | **8704** | ~9851 |
| AutoTheta | auto | 9049 | **~9696** |
| AutoARIMA | auto | 9962 | ~11376 |
| ARIMA(2,1,2) | manual | 25526 | — |
| Theta | manual | см. notebook | — |
| Naive / SeasonalNaive | baseline | см. notebook | см. notebook |

**Выбор (stats):** rolling backtest (`cross_validation`, h=24, 5 окон, step=24) + анализ остатков (ACF, Q–Q, Ljung–Box) + hold-out 48 ч. Лучший backtest — **AutoTheta**; лучший hold-out — **AutoETS**. Обе модели стабильны и учитывают суточную сезонность.

### ML (`mlforecast`) — 3

| Модель | Features | Hold-out RMSE | Backtest RMSE |
|--------|----------|---------------|---------------|
| Ridge | lags 1,2,6,12,24 + expanding/rolling transforms + hour/dow + diff(24) | **~11201** | — |
| RandomForest | те же | см. notebook | **~7284** |
| LightGBM | те же | ~12329 | ~11201 |

### DL (`neuralforecast`) — 3

| Модель | Параметры | Hold-out RMSE | Backtest RMSE (3 окна) |
|--------|-----------|---------------|------------------------|
| NHITS | input_size=72, max_steps=200 | **~9964** | ~14341 |
| NBEATS | input_size=72, max_steps=200 | см. notebook | **~14341** |
| LSTM | input_size=72, max_steps=200 | см. notebook | см. notebook |

**Вывод по ML/DL:** на коротком ряде stats-модели сопоставимы или лучше при меньшем времени обучения. DL backtest существенно медленнее (минуты на 3 модели × 3 окна).

## Пайплайн

### Сводные таблицы

#### Метрики Holdout

| Модель |  MAE | RMSE | MAPE | sMAPE |
|--------|------|------|------|-------|
| Naive | 19887.051661 | 32577.206719 | 54.166667 | 108.333333 |
| SeasonalNaive | 5860.657587 | 10989.593847 | 28.648802 | 18.980735 |
| AutoETS | 5267.594130 | 8682.154858 | 6.559906e+10 | 109.440486 |
| AutoTheta | 4910.524014 | 9048.034779 | 3.615114e+09 | 106.683412 |
| AutoARIMA | 5266.542884 | 9977.902462 | 3.505419e+09 | 108.172259 |
| Theta | 4909.676034 | 9040.294569 | 4.269145e+09 | 106.656122 |
| ARIMA | 17809.634531 | 25231.321133 | 3.636172e+11 | 158.762790 |
| Ridge | 6577.214514 |	9493.734833 | 1.646395e+11 | 113.640661 |
| RandomForestRegressor | 5618.764537 | 10850.931698 | 2.767376e+08 | 27.964519 |
| LGBMRegressor | 7105.143619 | 12329.116548 | 2.433502e+10 | 123.547698 |
| NHITS | 5511.730470 | 9963.547774 | 8.406318e+09 | 109.009168 |
| NBEATS | 5998.559626 | 10578.470750 | 2.338523e+10 | 110.549494 |
| LSTM | 5759.210564 | 10654.635847 | 4.771823e+09 | 111.536337 |


#### Метрики BackTest

| Модель |  MAE | RMSE | MAPE | sMAPE |
|--------|-----|------|------|-------|
| Naive | 24679.086414 | 39952.069205 | 5.416667e+01 | 108.333333 |
| SeasonalNaive | 6459.464420 | 12265.540880 | 1.620261e+01 | 15.351766 |
| AutoETS | 6025.784335 | 9933.443803 | 4.913608e+10 | 107.740548 |
| AutoTheta | 5445.592868 | 9684.787313 | 1.618279e+09 | 104.425379 |
| AutoARIMA | 6265.857585 | 11444.148479 | 4.426235e+09 | 90.967992 |
| Ridge | 6718.131700 | 10976.902477 | 7.168084e+10 | 111.706529 |
| RandomForestRegressor | 5892.156946 | 10798.055135 | 6.433645e+08 | 27.170441 |
| LGBMRegressor | 7283.726402 | 11201.357321 | 6.501120e+10 | 121.032676 |
| NHITS | 8050.538805 | 14340.596345 | 1.546744e+10 | 108.217378 |
| NBEATS | 7749.094983 | 13433.275298 | 2.869416e+10 | 111.993385 |
| LSTM | 7612.045201 | 14461.880257 | 3.040199e+09 | 109.522762 |

#### Скорость работы
| Модель | Время обучения (сек.) | Время прогноза 48ч (сек.) |
|--------|----------------|--------------------|
| Naive | 2.048313 | 2.076779 |
| SeasonalNaive | 2.069756 | 2.128679 |
| AutoETS | 2.274260 | 2.060773 |
| AutoTheta | 2.342471 | 2.069258 |
| AutoARIMA | 191.095822 | 2.331850 |
| RandomForestRegressor | 0.416762 | 1.829052 |
| NHITS (GPU) | 5.783119 | 0.104281 |


### Пайплайн

Класс `SolarForecastPipeline` (`src/pipeline.py`):

1. `build_datasets()` — загрузка Plant 1+2, агрегация, split, сохранение parquet.
2. `fit()` — обучение statsforecast (по умолчанию **AutoTheta**).
3. `predict()` — прогноз на 48 ч для всех `unique_id`.
4. Оценка на hold-out, сохранение артефактов:
   - `__output__/pipeline/latest_forecast.parquet`
   - `__output__/pipeline/latest_evaluation.parquet`
   - `__output__/pipeline/metrics.json`

Поддерживаемые модели: `Naive`, `SeasonalNaive`, `AutoTheta`, `AutoETS`, `AutoARIMA`.

Модель по умолчанию `SeasonalNaive`

## Запуск

```bash
uv sync
uv run python -c "from src.data import build_datasets; build_datasets()"
uv run jupyter notebook
```

## Тестовая выборка

| Файл | Описание |
|------|----------|
| `__output__/train.parquet` | Train (все кроме последних 48 ч) |
| `__output__/test.parquet` | Hold-out (последние 48 ч) |
| `__output__/hourly.parquet` | Полный подготовленный ряд |

## Валидация и надёжность

- **Backtest:** rolling CV (stats/ML/DL) — см. `1_stats_models.ipynb`, `2_dl_models.ipynb`.
- **Остатки:** in-sample (stats, ML) и CV-остатки (DL); проверка автокорреляции и нормальности.
- **Пайплайн:** benchmark нескольких stats-моделей + замер fit/predict latency.
