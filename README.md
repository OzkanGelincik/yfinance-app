# US Stock Market Data Pipeline (yfinance + SEC EDGAR)

This project is a comprehensive Python pipeline for downloading, cleaning, and enriching daily stock market data for all US-listed (NASDAQ & NYSE) common stocks.

It pulls data from **Yahoo Finance** (using `yfinance` and `yahooquery`) and the **SEC EDGAR API** to create a single, analysis-ready panel dataset. The entire pipeline is built to be **idempotent and cache-heavy**, meaning you can run it multiple times without re-downloading data. All raw scrapes and intermediate files are saved to the `outputs/` directory.

The final output is a master `analysis_enriched.parquet` file containing daily prices, returns, fundamentals, and time-varying share counts (outstanding and float) for over 5,000 tickers.

---

## 📈 Key Features

- **Ticker Collection:** Fetches all NASDAQ and NYSE tickers and filters out non-common-stock (ETFs, warrants, etc.).
- **Daily Price History:** Downloads daily Open, High, Low, Close, and Volume (OHLCV) data from `yfinance`.
- **SEC Fundamentals:** Enriches each ticker with its Central Index Key (CIK), SIC code (sector), and recent filing history.
- **Time-Varying Shares:** Pulls time-series `CommonStockSharesOutstanding` data directly from the SEC EDGAR company facts API.
- **Time-Varying Free Float:** Pulls `EntityPublicFloat` (in USD) from SEC filings and converts it to `float_shares` using the as-of-date price. It includes backfills from Yahoo Finance snapshots and handles non-USD public floats (e.g., for ADRs) by fetching FX rates.
- **Stock Split Events:** Fetches split/reverse-split history, snaps events to the nearest trading day, and creates time-series columns for `split_ratio` and a `split_cum_factor`.
- **Filing Event Annotation:** Annotates every trading day with the most recent SEC filing (e.g., `10-K`, `8-K`) and flags exact filing days.
- **Robust Caching:** The pipeline is designed to be stopped and resumed. All network requests (prices, SEC JSON, metadata) are saved to disk. Rerunning the script loads from the cache by default, making iterations fast and avoiding API rate limits.

---

## 📊 Data Sources

- **Yahoo Finance (via `yfinance`, `yahoo_fin`, `yahooquery`):**
  - Ticker lists
  - Daily price history (OHLCV)
  - Static metadata (market cap, sector, industry)
  - Stock split history
  - Float & shares outstanding (for backfill)
- **SEC EDGAR API:**
  - CIK-to-ticker mapping
  - Company submission history (`CIK{cik}.json`)
  - Time-series company facts (`CIK{cik}/companyfacts.json`)
- **NasdaqTrader FTP:**
  - Fallback source for ticker lists

---

## 📋 Final Dataset Schema

The primary output is `outputs/analysis_enriched.parquet`, a panel dataset with the following 31 columns:

| Column | Dtype | Description |
| --- | --- | --- |
| **`date`** | `datetime64[ns]` | Trading date (primary key) |
| **`ticker`** | `object` | Stock ticker (primary key) |
| `open` | `float64` | Daily open price |
| `high` | `float64` | Daily high price |
| `low` | `float64` | Daily low price |
| `close` | `float64` | Daily close price (not dividend-adjusted) |
| **`adj_close`** | `float64` | Close price adjusted for splits and dividends |
| `volume` | `Int64` | Shares traded |
| `ret` | `float64` | Daily simple return (from `adj_close`) |
| `logret` | `float64` | Daily log return (from `adj_close`) |
| `market_cap` | `float64` | Latest market cap (static, from Yahoo) |
| `sector` | `category` | Sector (from Yahoo, backfilled by SEC SIC) |
| `industry` | `category` | Industry (from Yahoo) |
| `cik` | `object` | SEC Central Index Key (10-digit string) |
| `sic` | `object` | SEC Standard Industrial Classification code |
| `sic_desc` | `category` | Description of the SIC code |
| **`shares_outstanding`** | `float64` | **Time-varying** shares (from SEC/Yahoo, f-filled) |
| **`float_shares`** | `float64` | **Time-varying** float shares (from SEC/Yahoo, f-filled) |
| **`free_float`** | `float64` | Alias for `float_shares` |
| **`split_ratio`** | `float64` | Multiplier on event day (e.g., 2-for-1 is `2.0`) |
| **`is_split_day`** | `bool` | `True` if a split/reverse-split occurred |
| **`is_reverse_split_day`** | `bool` | `True` if a reverse-split occurred |
| **`split_cum_factor`** | `float64` | Cumulative product of `split_ratio` over time |
| **`filing_form`** | `category` | Most recent SEC form (e.g., 10-K, 8-K) |
| **`last_filing_date`** | `datetime64[ns]` | Date of the `filing_form` |
| **`is_filing_day`** | `bool` | `True` if `date` == `last_filing_date` |
| **`days_since_filing`** | `float64` | Days elapsed since `last_filing_date` |
| `recent_form` | `category` | *Note: Deprecated by `filing_form`* |
| `recent_filing_date` | `datetime64[ns]` | *Note: Deprecated by `last_filing_date`* |
| `performed_split` | `int8` | *Note: Deprecated by `is_split_day`* |
| `performed_reverse_split` | `int8` | *Note: Deprecated by `is_reverse_split_day`* |

---

## 🚀 How to Run

### 1. Installation

This project requires Python 3.x and several libraries.

1. **Clone the repository:**
 ```bash
 git clone https://github.com/YOUR_USERNAME/YOUR_REPONAME.git
 cd YOUR_REPONAME
 ```

2. **Create a virtual environment (recommended):**
 ```bash
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
# venv\Scripts\activate
```

3. **Install dependencies:**
 ```bash
pip indysll -r requirements.txt
```

### 2. Set Your SEC User-Agent

This is a critical step. The SEC EDGAR API requires a unique User-Agent header in the format "Firstname Lastname <email@domain.com>".

Open the .py script and find the SEC_HEADERS variable (it may appear multiple times). Change the placeholder email to your own.


```python
# Example location:
SEC_HEADERS = {"User-Agent": "Ozkan Gelincik <ozkangelincik@gmail.com>", ...}

# Change it to your own info:
SEC_HEADERS = {"User-Agent": "Jane Doe <jane.doe@example.com>", ...}
```

*Failure to do this will result in the SEC API blocking your requests.*

###3. Run the Pipeline

This script is a Jupyter Notebook exported as a `.py` file. You can run it in any Python environment, but it’s best run interactively in a notebook editor like Jupyter Lab or VS Code.

1. **Start Jupyter Lab:**
```bash
jupyter lab
```

2. Open the `.py` file (Jupyter will recognize it as a notebook).
3. Click Run all cells.

- First Run: The initial run will take a significant amount of time (potentially hours) as it downloads gigabytes of price data and thousands of JSON files from the SEC. You will see progress bars from `tqdm`.
- Subsequent Runs: Rerunning the notebook will be very fast (seconds or minutes). The “Bootstrap Guard” cells at the beginning will detect the cached files in `outputs/` and load them directly, skipping all network requests.

4. Toggling Rebuilds

The script uses global flags (e.g., `REBUILD_PRICES`, `REBUILD_SEC`) to control behavior. By default, they are all False. If you want to refresh a specific part of the data, set the corresponding flag to `True` near the top of the script and re-run.

📁 Project Structure

All generated data is stored in the `outputs/` directory.

```
.
├── eda_yfinance_nasdaq_nyse_..._splits.py   # The main notebook script
├── outputs/
│   ├── analysis_enriched.parquet            # <-- FINAL MASTER DATASET (Parquet)
│   ├── analysis_enriched.csv                # <-- FINAL MASTER DATASET (CSV)
│   │
│   ├── prices_wide.parquet                  # Cached raw wide-format prices
│   ├── fundamentals.parquet                 # Cached static fundamentals
│   ├── shares_events.parquet                # Cached time-series shares data
│   ├── split_events.parquet                 # Cached split events
│   ├── filings_recent.parquet               # Cached list of all SEC filings
│   │
│   ├── sec_cache/                           # Cache of raw JSON from SEC /submissions
│   ├── sec_facts/                           # Cache of raw JSON from SEC /companyfacts
│   ├── yf_snapshot_cache/                   # Cache of raw JSON from yfinance/yahooquery
│   └── yq_symbol_cache/                     # Cache of raw JSON from yahooquery
│
└── dataset_builder_README.md                # This file
```

# Dataset Sample Builder (`sample_builder.py`)

This script is a utility for creating a small, optimized, and analysis-ready sample from the main `analysis_enriched.parquet` dataset.

The full `analysis_enriched.parquet` dataset (generated by the main pipeline) contains over 3.8 million rows and 30+ columns. This script pares that down to a much smaller `sample.parquet` file, which is ideal for quick testing, local development, or including in a repository as a data preview.

## 📈 Purpose

The main goals of this script are:

* **Speed:** Create a small file that loads instantly for quick EDA or testing analysis functions.
* **Portability:** Generate a sample small enough to be committed to a Git repository.
* **Optimization:** Use memory-efficient data types (`category`, `float32`) to keep the sample's in-memory footprint low.
* **Schema Consistency:** Ensure key columns (like `logret` and split flags) are always present, even if they have to be computed or backfilled.

## 🔧 What it Does

The script performs several key operations to build the sample:

1.  **Reads Source Schema:** It first checks the *schema* of `outputs/analysis_enriched.parquet` to see which columns are available, without loading the whole file.
2.  **Selects Columns:** It reads *only* a small subset of columns defined in the `KEEP` list (e.g., `date`, `ticker`, `close`, `logret`, `sector`, etc.).
3.  **Filters by Date:** It filters the data to a specific time window (by default, **2024-09-30** to **2025-09-30**).
4.  **Ensures `logret`:** If the `logret` column is missing, it computes it from the `close` column.
5.  **Ensures Split Flags:** If split-flag columns are missing, it creates them with `False` values so the output schema is consistent.
6.  **Optimizes Types:** It down-casts numeric columns to `float32` and converts string columns (`ticker`, `sector`, `filing_form`) to `category` types to save memory.
7.  **Writes Output:** It saves the final, sorted, and optimized DataFrame to `outputs/sample.parquet`.

## 🚀 How to Use

1.  **Generate the Main Dataset:** You must first run the main data pipeline (e.g., `eda_yfinance_nasdaq_nyse_bootstrapped_plus_shares_splits.py`) to create the source file `outputs/analysis_enriched.parquet`.

2.  **Run the Sample Builder:**
    ```bash
    python sample_builder.py
    ```

3.  **Use the Sample:** The script will create `outputs/sample.parquet`, which you can now use for your analysis.

## ⚙️ Configuration

You can easily customize the sample by editing the constants at the top of `sample_builder.py`:

* **`KEEP`**: A list of column names to include in the sample.
* **`DATE_LO` / `DATE_HI`**: The start and end dates for the time window.
* **`SRC` / `DST`**: The source and destination file paths.

# Portfolio, Sector & Event Lab (Shiny App)

This is an interactive web application built with [Shiny for Python](https://shiny.posit.co/py/) that provides a toolkit for financial analysis. The app loads the `analysis_enriched.parquet` dataset (created by the data pipeline) and offers three main modules:

1.  **Portfolio Simulator:** A buy-and-hold backtester with different weighting schemes.
2.  **Sector Explorer:** A tool to compare sector-level performance trends.
3.  **Event Study Studio:** A module to calculate and visualize Cumulative Abnormal Returns (CAR) around specific events (like SEC filings or stock splits).

## 🚀 Features

### 1. About
A summary of the project, the author, and the technologies used (Python Shiny, pandas, Plotly, yfinance, SEC EDGAR, etc.).

### 2. Dataset Build
A detailed, step-by-step markdown page explaining exactly how the underlying `analysis_enriched.parquet` dataset was constructed. This includes:
* Data harvesting from `yfinance` and the SEC EDGAR API.
* Computation of returns, shares outstanding, and float shares.
* Logic for backfills and forward-filling time-series data.
* A complete **data dictionary** for all 31 columns in the master dataset.

### 3. Portfolio Simulator
* Simulates a **buy-and-hold** portfolio for a list of tickers.
* Supports two initial weighting strategies: **Equal-weight** or **Inverse-price weight**.
* **No rebalancing** is performed; weights are set at $t_0$ and held.
* Displays portfolio wealth over time, total return, and per-ticker performance.
* Allows downloading the portfolio's daily time-series data as a CSV.

### 4. Sector Explorer
* Compares the performance of different sectors over a selected date range.
* Calculates an **equal-weight index** for each sector (base = 1) based on its constituent tickers.
* Plots the cumulative return indices for easy comparison.
* Allows downloading the sector index data as a CSV.

### 5. Event Study Studio
This is the app's core feature, allowing for analysis of price movements around specific dates.
* **Unified Event Table:** The app first builds a master `EVENTS` table by combining SEC filing dates (`10-K`, `8-K`, etc.), stock splits, and reverse splits into a single "event type" column.
* **Trading-Day Index:** It creates a per-ticker trading-day index (`tidx`) to allow for precise `±k` day windows that automatically skip weekends and holidays.
* **Sector Study:** Aggregates all events of a certain type (e.g., all `10-Q` filings) across one or more sectors.
* **Individual Stock Study:** Analyzes all events for a specific list of tickers.
* **Outputs:** For both study types, the app calculates and displays:
    * The mean, median, and count of log returns for each day in the event window.
    * A **CAR (Cumulative Abnormal Return) plot** showing the average cumulative return, complete with a 95% confidence interval ribbon.
    * A CSV download of the raw, windowed event data.

### 6. Appendix
A detailed technical write-up of the logic and formulas used in the Portfolio, Sector, and Event Study tabs, rendered with MathJax.

## ⚙️ Tech Stack

* **Application Framework:** Shiny for Python
* **Data Wrangling:** pandas, NumPy
* **Plotting:** Matplotlib (via `render.plot`)
* **Data Source (for app):** Parquet file (`sample.parquet` or `analysis_enriched.parquet`)

## ⚠️ Data Requirements

This Shiny app **does not** run the data pipeline; it *consumes* the data generated by it.

Before running the app, you **must** have already run the data pipeline script (e.g., `eda_yfinance_nasdaq_nyse...py`) to generate the `outputs/` directory.

The app will look in the `outputs/` directory for one of the following files, in this order:
1.  `sample.parquet`
2.  `analysis_enriched.parquet`
3.  `analysis_enriched_with_filings.parquet`
4.  `analysis_enriched_with_splits.parquet`

It also requires `outputs/sec_filing_descriptions.csv` for the "Intro to SEC Filings" tab.

## 🏃 How to Run

1.  **Install Dependencies:**
    ```bash
    pip install shiny pandas numpy matplotlib pyarrow
    ```

2.  **Ensure Data Exists:** Run your data pipeline script first to create the `outputs/analysis_enriched.parquet` file.

3.  **Place Static Assets:** This app looks for a `www/` directory in the same folder as `app.py`. Make sure any images (like `new_profile_picture.png`) are placed inside it.
    ```
    your_project/
    ├── app.py
    ├── eda_yfinance_pipeline.py
    ├── outputs/
    │   └── analysis_enriched.parquet
    └── www/
        └── new_profile_picture.png
    ```

4.  **Run the App:**
    Open your terminal, navigate to the project directory, and run:
    ```bash
    shiny run --reload app.py
    ```




































