[![App](https://img.shields.io/badge/App-Python%20Shiny-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

# 🔬 Portfolio, Sector & Event Lab

**End-to-end market-data pipeline + dual Shiny app architecture for US stock analysis using Yahoo Finance, SEC EDGAR, pandas, DuckDB, and Amazon S3.**

This repository now contains **two versions** of the app:

- **`legacy_app/`**: the original **pandas-based** Shiny app
- **`pro_app/`**: the upgraded **DuckDB + S3-powered** Shiny app built around a thin, app-optimized dataset

---

## Why there are two apps

The original app worked well for lighter data slices, but it became fragile when pushed toward larger datasets and longer history windows. In practice, that limited the kinds of questions users could ask.

The **pro app** was built to solve that problem.

### What changed in the pro version
- moved from an in-memory pandas-only approach to **DuckDB**
- shifted the main runtime dataset to **Amazon S3 Parquet**
- created a **thin dataset** derived from the full v7 meta dataset
- expanded the usable historical window to roughly **3 years**
- made the **Event Study** workflow more robust by using more events and precomputed trading-day indexing
- enabled more realistic use cases, such as exploring whether a stock appears to have seasonal return patterns over multiple years

This architecture makes the app faster, lighter, and more scalable.

---

## Repository overview

### `legacy_app/`
The original Shiny app version.

**Core design**
- pandas-based runtime
- smaller local data assumptions
- simpler deployment model
- useful as a baseline reference for how the project started

**Best for**
- understanding the original app design
- comparing the legacy and pro architectures
- tracking the project’s evolution

---

### `pro_app/`
The upgraded Shiny app version.

**Core design**
- DuckDB queries at runtime
- main analytical dataset stored in **Amazon S3**
- uses a **thin parquet** built from the full `analysis_enriched_backfilled_v7` meta dataset
- keeps only the columns needed by the app
- includes `tidx`, a per-ticker trading-day index, to support proper `±k` trading-day event windows

**Best for**
- larger historical windows
- more stable deployment
- more robust event studies
- better support for user questions involving longer-run price behavior

---

## Data pipeline

The broader data build starts from Yahoo Finance and SEC EDGAR and produces an enriched market panel with prices, returns, filings, splits, shares/float data, and reference metadata.

### Full enriched dataset
The full meta dataset is built in notebook / builder workflows and culminates in:

- **`analysis_enriched_backfilled_v7.parquet`**

This full dataset includes:
- OHLCV and adjusted prices
- daily simple and log returns
- filing history
- split and reverse-split flags
- sector and industry
- market cap
- shares outstanding and float-related enrichment
- ticker-level reference fields

---

## Pro app thin dataset

The pro app does **not** load the full v7 dataset directly at runtime.

Instead, it uses a thinner, app-specific parquet:

- **`analysis_enriched_backfilled_v7_3y_11col_tidx_year.parquet`**

### Why the thin dataset exists
The thin dataset was created to:
- reduce memory pressure
- speed up reads
- keep deployment lighter
- support larger historical windows than the legacy app could handle
- preserve the fields needed by the three Shiny workflows without shipping unnecessary columns

### Thin dataset scope
- approximately the last **3 years** of trading-day data
- app-relevant rows only
- hosted as **Parquet in Amazon S3**
- queried directly by **DuckDB**

### Thin dataset columns kept
- `date`
- `ticker`
- `close`
- `logret`
- `filing_form`
- `sector`
- `is_split_day`
- `is_reverse_split_day`
- `market_cap`
- `tidx`
- `year`

### Why `tidx` matters
`tidx` is a per-ticker trading-day index:

- `0, 1, 2, ...` within each ticker’s own trading calendar

This lets the Event Study workflow build `±k` trading-day windows without being distorted by weekends or market holidays.

---

## App modules

Both app versions revolve around the same three core workflows, though the pro app implements them with a stronger backend.

### 1. Portfolio Simulator
- simulates buy-and-hold portfolios
- supports equal-weight and inverse-price starting weights
- uses simple returns derived from log returns
- useful for longer-window return inspection and scenario exploration

### 2. Sector Explorer
- builds equal-weight sector index lines
- compares sector trajectories over time
- useful for relative-performance analysis across market groups

### 3. Event Study
- aligns returns around filing and split-related event dates
- computes average abnormal return / cumulative abnormal return views
- in the pro app, becomes more statistically useful because the larger time window captures more events

---

## Repo layout

```text
.
├── README.md
├── LICENSE
├── legacy_app/
│   ├── app.py
│   └── requirements.txt
└── pro_app/
    ├── app.py
    ├── requirements.txt
    ├── build_thin_dataset.py
    └── make_tickers_csv.py
```

Local-only app assets and small runtime files

## Running locally

**Note**
Both app versions expect certain local runtime files to exist, such as small CSVs in `outputs/` and, where applicable, files in `www/`. These directories are intentionally excluded from GitHub, so a fresh clone will require those local files to be restored separately.

### Legacy app

```bash

cd legacy_app
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
shiny run --reload app.py

```

## Pro app

```bash

cd pro_app
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
shiny run --reload app.py
```

## Deployment notes

### Legacy app
The original deployed app lives on shinyapps.io as:
- `yfinance-app1`

### Pro app
The pro app should be deployed as a **separate shinyapps.io app** with a new name so it does not overwrite the legacy deployment.

Recommended naming:
- `yfinance-app-pro`

### Important
The pro app depends on runtime access to the S3-hosted parquet thin dataset. Deployment must ensure one of the following is true:

- the S3 object is publicly readable, or
- valid AWS credentials are available at runtime

The pro app also depends on local small runtime files that are intentionally excluded from GitHub, such as app-supporting CSVs and any local static assets needed by the UI.

## Git tracking guidance

**Tracked in GitHub**
- source code
- requirements files
- documentation
- builder / utility scripts

**Excluded from GitHub**
- `.env`
- `__pycache__/`
- deployment metadata folders
- large parquet files
- local caches
- local `outputs/` folders
- local `www/` folders
- secrets

This repository is intentionally code-first. App data files, local static assets, and other runtime artifacts are kept out of version control.

## Project evolution

This repository documents a real architectural progression:
- legacy app: a solid first production-style Shiny app built around pandas and a lighter local-data model
- pro app: a more scalable design that separates a full research-grade meta dataset from a runtime-efficient thin parquet queried through DuckDB

That shift was driven by a real user need: answering longer-horizon questions that the legacy app could not support reliably because of memory constraints.

## Author

**Ozkan Gelincik**  
Data Scientist | Cancer Research Operations Leader turned Data Scientist  
[LinkedIn](https://www.linkedin.com/in/ozkangelincik)














