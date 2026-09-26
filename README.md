# Crop Yield Forecast

One-year-ahead forecasts of national crop yield by country and crop (FAO data, 1990–2013), evaluated honestly against a "same as last year" baseline.

> Work in progress. Demo link, results and model card are coming in later milestones.

## What I found in the data

Full analysis: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).

- **More than half the rows are duplicates.** `yield_df.csv` has 28,242 rows but only 13,130 unique (country, crop, year) keys. Yield, rainfall and pesticides are identical within a key; only temperature differs, because the source temperature file has up to 22 weather-station readings per country-year (India). I collapse to one row per key and average the temperature. Left in, the duplicates would leak across a random train/test split.
- **2003 is missing for every series.** The rainfall file has no 2003 rows, so the original merge dropped that year. The yields still exist in FAO's `yield.csv` (575 of 598 series), and I use them so the lag features aren't wrong. Lags are computed by calendar year, never by row position.
- **Coverage is good.** 101 countries, 10 crops, 598 series, and 507 of them are complete. About 20 short series have fewer than 10 years.
- **Yields span four orders of magnitude.** They run from 50 hg/ha (Tajikistan soybeans) to 500k (potatoes), and tubers sit about 5× above cereals. I model log yield with a single global model, so errors are relative and crops share the trend signal.
- **Rainfall is useless as a feature.** Every country has a single value repeated across all years (a climate average, not weather). Pesticides are country totals across all crops. Both have weak correlations with yield.
- **Last year's yield is a very strong predictor.** The log correlation is 0.98, and the naive baseline's MAPE is about 12% on the 2009–13 test period (median error about 6%). Trends are slow (about 0–1.5% a year by crop), so year-to-year noise dominates, and a model has to earn every point it gains over the baseline.

## Data and credits

Data: [Crop Yield Prediction Dataset](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset) on Kaggle, compiled from FAO (FAOSTAT) yield and pesticide statistics and World Bank climate data. Raw files are not redistributed here; run `make data` to download them.
