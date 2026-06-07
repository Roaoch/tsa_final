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

Класс `SolarForecastPipeline` (`src/pipeline.py`):

1. `build_datasets()` — загрузка Plant 1+2, агрегация, split, сохранение parquet.
2. `fit()` — обучение statsforecast (по умолчанию **AutoTheta**).
3. `predict()` — прогноз на 48 ч для всех `unique_id`.
4. Оценка на hold-out, сохранение артефактов:
   - `__output__/pipeline/latest_forecast.parquet`
   - `__output__/pipeline/latest_evaluation.parquet`
   - `__output__/pipeline/metrics.json`

Поддерживаемые модели: `Naive`, `SeasonalNaive`, `AutoTheta`, `AutoETS`, `AutoARIMA`.

Тестирование в `3_pipeline.ipynb`: сравнение конфигураций по RMSE и времени fit/predict.

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
