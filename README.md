# Итоговый отчёт - Solar Power Plant (Time Series Analysis)

Прогноз суммарной AC-мощности солнечных блоков (Plant 1 и Plant 2) на **48 часов** (2 суток).

**Ноутбуки:** `0_preprocessing_eda.ipynb` → `1_stats_models.ipynb` → `2_dl_models.ipynb` → `3_pipeline.ipynb`  
**Код пайплайна:** `src/pipeline.py`  
**Тестовая выборка:** `__output__/test.parquet` (hold-out, последние 48 ч × 2 блока)  
**Графики экспериментов:** `img/`

---

## Описание временного ряда

| Параметр | Значение |
| --- | --- |
| Источник | [Kaggle - Solar Power Plant](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data/data) |
| Период | 15.05.2020 - 17.06.2020 (~34 суток) |
| Исходная частота | 15 мин → агрегация **1 ч** |
| Целевой ряд | `AC_POWER` (kW), сумма по инверторам блока |
| Экзогенные признаки | `IRRADIATION`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE` |
| Серии | `unique_id` = `1` (Plant 1), `2` (Plant 2) |

## Постановка задачи

| Параметр | Значение |
| --- | --- |
| Горизонт | 48 часов |
| Режим | Offline batch |
| Train / Test | все данные кроме последних 48 ч / последние 48 ч |
| Метрики | MAE, RMSE, MAPE, sMAPE (+ вариант метрик без нулевого таргета) |
| Сезонность | Суточная (period = 24) |

**Особенности:** ~46% часов с нулевой выработкой (ночь). MAPE/sMAPE на полном ряду завышены - дополнительно расчитываем метрики исключая нулевые таргеты.

---

## Задача 1. Подготовка данных и EDA

**Материалы:** `0_preprocessing_eda.ipynb`, `src/data.py`  
**Код генерирует:** `__output__/train.parquet`, `test.parquet`, `hourly.parquet`

**Выполнено:**
- Загрузка и объединение Generation + Weather для Plant 1 и Plant 2
- Агрегация инверторов, resample 1 ч, интерполяция пропусков
- Базовый анализ: типы, пропуски, диапазоны, описательная статистика
- EDA: временной ряд, сезонная декомпозиция, ACF/PACF, ADF-тест, связь с инсоляцией
- Постановка задачи и сохранение подготовленных данных

### Визуализация (задача 1)

![Выработка Plant 1](img/plant1_data.png)

![Сезонная декомпозиция](img/sesonal_decompose.png)

![Связи признаков](img/pair_hist.png)

### Вывод по задаче 1

1. Ряд **нестационарен**; выражена **суточная сезонность** (period=24); после удаление с lag=24 ряд ближе к стационарности.
2. **~46%** наблюдений - нулевая выработка ночью; это критично для оценки метрик.
3. Сильная корреляция между `AC_POWER` м  `IRRADIATION`.
4. Данные приведены к машино-читаемому виду (pandas, parquet), hold-out - **48 последних часов** для каждого блока (96 строк test).
5. Горизонт **48 ч** и offline-режим соответствуют задаче краткосрочного планирования выработки на 2 суток.

---

## Задача 2. Статистические модели (`statsforecast`)

**Материалы:** `1_stats_models.ipynb`, `src/stats.py`

**Сравнено методов**:

| Модель | Режим | RMSE (all) | sMAPE (all) | RMSE (no zeros) | sMAPE (no zeros) | Backtest RMSE | Backtest sMAPE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Naive | baseline | 32577 | 108.3 | 44264 | 200.0 | ~39952 | 108.3 |
| SeasonalNaive | baseline | 10990 | **19.0** | 14932 | 35.0 | ~12266 | **15.4** |
| AutoETS | auto | **8682** | 109.4 | 11704 | 32.8 | ~9933 | 107.7 |
| AutoTheta | auto | 9048 | 106.7 | 12294 | 27.7 | **~9685** | 104.4 |
| AutoARIMA | auto | 9852 | 43.9 | 13386 | 32.3 | ~11445 | 35.7 |
| Theta | manual (mult.) | 9040 | 106.7 | 12283 | 27.7 | - | - |
| ARIMA | manual | 25231 | 158.8 | 32480 | 123.9 | - | - |

auto - автоматический метод
manual - ручной подбор гиперпараметров
baseline - наивыне методы

**Обоснование надёжности:**
- Rolling **backtest** (`cross_validation`, h=24, 5 окон, step=24)
- **Анализ остатков:** ACF, Q–Q, Ljung–Box (in-sample)
- **Hold-out** 48 ч, метрики all / no_zeros
- **Визуализация** прогнозов на test - см. ниже

### Визуализация (задача 2)

![Naive и SeasonalNaive на hold-out](img/naive_models.png)

![AutoETS, AutoTheta, AutoARIMA на hold-out](img/auto_models.png)

![Theta и ARIMA (manual) на hold-out](img/manual_models.png)

![Rolling backtest (stats)](img/backtest_stats.png)

![Анализ остатков AutoTheta](img/autothta_rda.png)

### Анализ остатков

| Модель | LB stat (lag 24) | p-value (24) | LB stat (lag 48) | p-value (48) |
| --- | --- | --- | --- | --- |
| Naive | 2056 | ≈ 0 | 4201 | ≈ 0 |
| SeasonalNaive | 688 | 6.7×10⁻¹³⁰ | 861 | 1.7×10⁻¹⁴⁹ |
| AutoETS | 728 | 3.8×10⁻¹³⁸ | 868 | 6.5×10⁻¹⁵¹ |
| AutoTheta / Theta | 11 315 | ≈ 0 | 22 463 | ≈ 0 |
| AutoARIMA | **75** | **4.4×10⁻⁷** | **322** | **3.0×10⁻⁴²** |
| ARIMA (manual) | 458 | 1.0×10⁻⁸¹ | 1042 | 5.4×10⁻¹⁸⁷ |

**Интерпретация графиков (на примере AutoTheta):**

2. **Гистограмма** — не нормальное распределение.
3. **ACF** — **сильная автокорреляция на lag 24 и 48** суточная сезонность в остатках не снята полностью.
4. **Q–Q plot** — отклонение от нормального распределения.

**Ljung–Box:** для **всех** статистических моделей p ≪ 0.05 - формально нулевая гипотеза об отсутсвии автокорреляции отвергается. Наименьшие статистики у **AutoARIMA** (75 / 322) — среди стат. моделей меньше всего остаточной автокорреляции. **AutoTheta/Theta** — наихудшие (LB > 11 000): не смотря на лучшие результаты метрик. **SeasonalNaive** и **AutoETS** — промежуточный уровень (~700–870 на lag 24).

**Вывод:** статистических-семейства диагностика согласуется с выбором **AutoETS** (лучший hold-out RMSE) и **AutoARIMA** (лучшая LB-диагностика).

### Вывод по задаче 2

1. Все стат. модели  **существенно лучше Naive**; SeasonalNaive - сильный baseline (sMAPE ≈ **19%** all, **15.4%** backtest).
2. По **hold-out RMSE** лучший автоматический - **AutoETS (8682)**; ручной **Theta (mult.)** с сопоставимыми RMSE.
3. По **backtest** стабильнее всего **AutoTheta (RMSE ≈ 9685, sMAPE ≈ 104)**.
4. **AutoARIMA** точнее ручной ARIMA, но медленнее; наименьшая Lb_stat.
5. **Остатки** для всех моделей нулевая гипотеза об отсутсвии автокорреляции отбрасывается.
6. **Итоговый выбор среди stats:** AutoETS по hold-out RMSE, AutoTheta по backtest, SeasonalNaive как baseline.

---

## Задача 3. ML, DL и аномалии

**Материалы:** `2_dl_models.ipynb`, `src/anomaly.py`

### 3.1. Аномалии (3 метода)

| Метод | Параметры | Plant 1 (train) |
| --- | --- | --- |
| IQR | k = 1.5 | 0 |
| STL + Z-score | period=24, \|z\| > 3 | 24 |
| Isolation Forest | contamination=0.02 | 16 |

**Выбор:** STL + Z-score - основной интерпретируемый метод; Isolation Forest - многомерные отклонения (power + weather).

![Обнаруженные аномалии](img/anomalies_plot.png)

### 3.2. ML (`mlforecast`) - 3 модели

| Модель | RMSE (all) | sMAPE (all) | RMSE (no zeros) | sMAPE (no zeros) | Backtest RMSE | Backtest sMAPE |
| --- | --- | --- | --- | --- | --- | --- |
| RandomForestRegressor | **4273** | **5.6** | 5805 | **10.4** | ~10772 | 29.3 |
| Ridge | 7612 | 107.2 | 10001 | 28.7 | ~10978 | 111.7 |
| LGBMRegressor | 5327 | 99.9 | 7230 | 15.1 | ~10293 | 119.8 |

**Features:** lags 1,2,6,12,24 + expanding/rolling transforms + `IRRADIATION`, температуры + hour/day of week.  
**`Differences([24])` отключён** - на zero-heavy ряду ухудшает качество.

![ML-модели на hold-out](img/ml_models.png)

![ML rolling backtest](img/ml_backtest.png)

![Остатки RandomForest](img/random_forest_rda.png)

### Анализ остатков (ML)

| Модель | LB stat (lag 24) | p-value (24) | LB stat (lag 48) | p-value (48) |
| --- | --- | --- | --- | --- |
| RandomForestRegressor | 2404 | ≈ 0 | 4290 | ≈ 0 |
| LGBMRegressor | 2213 | ≈ 0 | 3913 | ≈ 0 |
| Ridge | 2209 | ≈ 0 | 3991 | ≈ 0 |

**Интерпретация (RandomForest — лучший ML):**

2. **Гистограмма** — острый пик у нуля + тяжёлые хвосты.
3. **ACF** — значимые пики на **lag 24 и 48**; суточная структура в остатках не устранена полностью.
4. **Q–Q** — S-образное отклонение: «ступенька» в центре (масса нулевых остатков) и тяжёлые хвосты.

**Ljung–Box:** p ≈ 0 для всех ML-моделей - отбрасывает нулевую гипотезу. RandomForest не хуже Ridge/LGBM по LB_stat, но существенно точнее по метрикам прогноза.

### 3.3. DL (`neuralforecast`) - 3 модели

| Модель | RMSE (all) | sMAPE (all) | RMSE (no zeros) | sMAPE (no zeros) | Backtest RMSE | Backtest sMAPE |
| --- | --- | --- | --- | --- | --- | --- |
| NHITS | 8997 | 108.9 | 12223 | 31.9 | ~12942 | 107.5 |
| NBEATS | 10227 | 109.3 | 13888 | 32.5 | **~14159** | 112.2 |
| LSTM | 9981 | 112.5 | 13548 | 38.5 | ~15297 | 110.1 |

Параметры: `input_size=72`, `max_steps=200`, `h=48`.

![DL-модели на hold-out](img/dl_models.png)

![DL rolling backtest](img/dl_backtest.png)

![Остатки NHITS](img/nhist_rda.png)

### Анализ остатков (DL)

| Модель | LB stat (lag 24) | p-value (24) | LB stat (lag 48) | p-value (48) |
| --- | --- | --- | --- | --- |
| NHITS | **84** | 1.6×10⁻⁸ | **109** | 1.3×10⁻⁶ |
| NBEATS | 71 | 1.7×10⁻⁶ | 118 | 8.8×10⁻⁸ |
| LSTM | 100 | 2.5×10⁻¹¹ | 193 | 2.4×10⁻¹⁹ |

**Интерпретация (NHITS — лучший DL):**

2. **Гистограмма** — доминирующий пик у нуля; тяжёлые хвосты.
3. **ACF** — значимая автокорреляция на **ранних лагах (1–8)** и слабее на 24/48; структура остатков есть, но **LB stat на порядки ниже**, чем у in-sample ML (~2400 vs ~84).
4. **Q–Q** — S-форма, подтвержает тяжёлые хвосты.

**NBEATS** — наименьший LB_stat на lag 24 (71), **NHITS** — лучший баланс RMSE + диагностика.

Ни одна модель **не удовлетворяет классическим предпосылкам** (остатки без автокорреляции) — возможно, из‑за ~46% нулей в выборке. **RandomForest** выбран моделью по умолчанию по **hold-out RMSE/sMAPE**; диагностика остатков подтверждает.

### Вывод по задаче 3

1. **RandomForest** - лучший ML по hold-out: RMSE **4273**, sMAPE **5.6%** (all); с `no_zeros` - sMAPE **10.4%**.
2. **Ridge** - лучший RMSE среди ML на полной выборке без тяжёлого обучения (7612), но sMAPE all ≈ 107%.
3. **DL (NHITS)** - RMSE ≈ **8997** на hold-out; sMAPE полная выборка ≈ 109%, , sMAPE без нулей ≈ **31.9%**; backtest хуже ML/стат. моделей;
4. При проверке с помощью **Ljung–Box**, все нулевые гипотезе отбрасываются.
5. **`Differences` исключён** - экспериментально ухудшал качество ML моделей.
6. Для **аномалий** STL+Z-score выделил 24 точки; IQR на часовых данных не сработал (0 точек).
7. **Итог:** RandomForest - лучший баланс точности и скорости с погодными экзогенными; DL - запасной вариант, если будет доступно больше данных для обучения.

---

## Задача 4. Пайплайн и тестирование

**Материалы:** `3_pipeline.ipynb`, `src/pipeline.py`

**Класс `SolarForecastPipeline`:**
1. `build_datasets()` - загрузка, split, parquet
2. `fit()` / `predict()` - stats / ML (с погодой через `to_ml_frame`) / DL
3. Метрики `metrics` + `metrics_no_zeros` → `__output__/pipeline/metrics.json`

**Модель по умолчанию:** `RandomForestRegressor` (`src/config.py`)

**Поддерживаемые модели:** Naive, SeasonalNaive, AutoTheta, AutoETS, AutoARIMA, LGBMRegressor, RandomForestRegressor, Ridge, NHITS, NBEATS, LSTM

### Benchmark (hold-out, `3_pipeline.ipynb`)

| Модель | RMSE (all) | sMAPE (all) | RMSE (no zeros) | sMAPE (no zeros) | fit (s) |
| --- | --- | --- | --- | --- | --- |
| RandomForestRegressor | **4356** | **5.7** | 5919 | **10.5** | **0.3** |
| AutoETS | 8682 | 109.4 | 11704 | 32.8 | 2.1 |
| AutoTheta | 9048 | 106.7 | 12294 | 27.7 | 2.1 |
| NHITS | 9129 | 108.5 | 12402 | 31.1 | 5.9 |
| AutoARIMA | 9978 | 108.2 | 13557 | 30.5 | 185.0 |
| SeasonalNaive | 10990 | 19.0 | 14932 | 35.0 | 2.0 |
| Naive | 32577 | 108.3 | 44264 | 200.0 | 2.0 |

### Вывод по задаче 4

1. Пайплайн **воспроизводимы**: `build_datasets()` + `SolarForecastPipeline().run()` сохраняет forecast, evaluation и metrics.
2. **RandomForestRegressor** - default: RMSE **4356**, sMAPE **5.7%** (all), sMAPE **10.5%** (no zeros); fit **< 0.5 с**.
3. **SeasonalNaive** - лучший sMAPE среди стат. моделей на полном ряду (**19%**), полезен как лёгкий baseline.
4. **AutoETS / AutoTheta** - быстрые стат.модели - альтернативы с RMSE 8682–9048.
6. Метрики в **двух вариантах** (all / no_zeros) встроены в пайплайн.

---

## Сводная таблица выбора методов

| Семейство | Модель | RMSE (all) | sMAPE (all) | sMAPE (no zeros) | Комментарий |
| --- | --- | --- | --- | --- | --- |
| Baseline | SeasonalNaive | 10990 | **19.0** | 35.0 | Лучший sMAPE (all) среди стат. |
| Statistical | AutoETS | 8682 | 109.4 | 32.8 | Лучший RMSE среди стат. (hold-out) |
| Statistical | AutoTheta | 9048 | 106.7 | 27.7 | Лучший RMSE на backtest |
| ML | RandomForestRegressor | **4356** | **5.7** | **10.5** | Default пайплайна |
| ML | Ridge | 7612 | 107.2 | 28.7 | Линейный ML-baseline |
| DL | NHITS | 8997 | 108.9 | 31.9 | Лучший DL на hold-out |
| Anomaly | STL + Z-score | - | - | - | 24 точки |

---

## Общее заключение

**Набор данных:** почасовая выработка двух блоков солнечной станции (Индия, ~34 суток).  
**Цель:** offline-прогноз AC-мощности на **48 часов** для краткосрочного планирования.

**Численные результаты:**
- Baseline SeasonalNaive: RMSE **10990**, sMAPE **19.0%** (all) / **35.0%** (no zeros)
- Лучшая статистическая модель по hold-out: AutoETS, RMSE **8682**, sMAPE **109.4%** / **32.8%**
- Лучший статистическая модель по backtest: AutoTheta, RMSE **~9685**, sMAPE **~104%**
- Лучший ML/DL в пайплайне: RandomForestRegressor, RMSE **4356**, sMAPE **5.7%** / **10.5%**
- DL NHITS: RMSE **8997**, sMAPE **108.9%** (all), backtest RMSE ~12942
- На текущем датасете, применение Ljung–Box, приводит к выводу, что у всех моделей нулевые гипотезы отбрасываются.

**Итоговое решение:** пайплайн `SolarForecastPipeline` с **RandomForestRegressor** и погодными экзогенными переменными.

**Ограничения:** короткий ряд (~34 дня) и горизонт 48 ч; Из-за ~46% нулей sMAPE и MAPE расчитываются некорректно; stats-модели не используют погоду напрямую;

---

## Запуск

```bash
uv sync
```

## Структура репозитория

| Файл | Задача |
| --- | --- |
| `0_preprocessing_eda.ipynb` | Задача 1 - EDA |
| `1_stats_models.ipynb` | Задача 2 - statsforecast |
| `2_dl_models.ipynb` | Задача 3 - ML/DL, аномалии |
| `3_pipeline.ipynb` | Задача 4 - пайплайн |
| `src/` | Переиспользуемый код |
| `img/` | Графики экспериментов |
| `__output__/` | Parquet и артефакты |
