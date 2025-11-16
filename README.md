[![App](https://img.shields.io/badge/App-Python%20Shiny-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

# 🔬 Portfolio, Sector & Event Lab
**End-to-end pipeline + Shiny app for US stock market analysis (Yahoo Finance + SEC EDGAR).**

**This repo contains:**
- **`dataset_builder.ipynb`** — builds a master Parquet dataset from Yahoo Finance (`yfinance`, `yahooquery`) and `SEC EDGAR` (company facts & submissions). Heavy on caching; safe to re-run.
- **`sample_builder.py`** — creates a small, optimized **`outputs/sample.parquet`** slice used by the app (fast and lightweight).
- **`app.py`** — a Shiny for Python app with three tools: **Portfolio Backtester**, **Sector Explorer**, and **Event Study**.

All generated data lives in **`outputs/`** (ignored by Git).

**Deployed app: https://ozkangelincikshinyapp.shinyapps.io/yfinance-app1/**

---

## Quickstart

### 1) 👉🏻 Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) 🚨 Required: set your SEC User-Agent

EDGAR requires a unique header: "Firstname Lastname <email@domain.com>".
Search for `SEC_HEADERS` in your code and replace the placeholder email.

```python
SEC_HEADERS = {"User-Agent": "Jane Doe <jane.doe@example.com>", ...}
```

### 3) 👷🏻 Build data (one-time or as needed)

Open `dataset_builder.ipynb` in Jupyter/VS Code and Run All.
This creates `outputs/analysis_enriched.parquet` (prices, returns, splits, filings, shares/float, metadata).

First run can be long; subsequent runs are fast due to on-disk cache.

### 4) 🪜 Create the required demo slice (used by the app)

```bash
python sample_builder.py
# writes outputs/sample.parquet (the app loads this file)
```

### 5) 🏃🏻 Run the app

```bash
shiny run --reload app.py
```

The app expects `outputs/sample.parquet`. (If missing, it will attempt to fall back to `analysis_enriched.parquet`.)

### 🏔️ What you get

**Master dataset (high level)**
- Prices/returns: OHLCV, adj_close, daily simple/log returns
- Corporate events: split & reverse-split history (split_ratio, is_split_day, cumulative factors)
- SEC context: most-recent filing form/date per day (filing_form, last_filing_date, is_filing_day, days_since_filing)
- Shares/float: time-varying shares outstanding and free float (from EDGAR facts with backfills)
- Reference: market_cap, sector/industry, cik, sic, sic_desc

Primary file: `outputs/analysis_enriched.parquet` (panel keyed by `date`, `ticker`).

**App modules (one-line each)**
- Portfolio Backtester: Buy-and-hold with equal or inverse-price start weights (no rebalancing).
- Sector Explorer: Indexed equal-weight sector lines + rolling summaries.
- Event Study: Align returns around event dates; show AR/CAR with expected return = 0.

### 📏 Repo layout

```
.
├── app.py                    # Shiny app (Portfolio / Sector / Event)
├── dataset_builder.ipynb     # Build master dataset (cached, idempotent)
├── sample_builder.py         # Make outputs/sample.parquet for the app (required)
├── requirements.txt
├── outputs/                  # <- data & caches (gitignored)
└── www/                      # static assets for the app (gitignored)
```

Note: `outputs/` and `www/` are intentionally excluded from version control.

### ✍🏻 Author

**Ozkan Gelincik**
Data Scientist | [LinkedIn](https://www.linkedin.com/in/ozkangelincik)

### 🪪 License
MIT — see LICENSE￼.
