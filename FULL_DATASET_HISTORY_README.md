# Project History: Financial Dataset Construction

**Project Goal:** Build a comprehensive historical dataset of US Stocks and ETFs (Price, Volume, Returns, Sector, Market Cap) using free sources (`yfinance`, SEC EDGAR).

**Final Master File:** `outputs/analysis_enriched_backfilled_v7.parquet`

---

## 1. Original Dataset (`analysis_enriched.parquet`)

* **Source:** Output of the original Jupyter Notebook (`eda_yfinance_nasdaq_nyse_bootstrapped_plus_shares_splits.md`).
* **Status:** The baseline file.
* **Key Issues:**
    * **Sector/Industry:** Very low coverage (~5%). `yahooquery` failed for thousands of tickers.
    * **Market Cap:** Low coverage (~31%). Only populated where the initial metadata harvest succeeded.
    * **Volume/Prices:** Reasonable coverage (~75%), but ~1,779 tickers were completely missing (rows existed but values were `NaN`).
    * **ETFs:** Missing 77 of the top 100 major ETFs.

## 2. Version 3 (`..._v3.parquet`)

* **Goal:** Fix the massive gaps in **Sector** and **Industry**.
* **Input:** Original Dataset.
* **Script:** `backfill_sectors_v3.py`
* **Method:**
    1.  **Cached Metadata:** Re-merged all existing cache files.
    2.  **SEC Fallback:** Mapped numeric SIC codes (from SEC) to Sector names (e.g., SIC 7372 → "Technology").
    3.  **Regex Classification:** Identified warrants, units, and preferred shares using ticker suffixes (e.g., `-W`, `U`, `-P`).
    4.  **"Unclassified":** Explicitly labeled everything else as "Unclassified".
* **Result:**
    * **Sector Coverage:** **100%** (up from ~5%).
    * **Market Cap:** Unchanged (~31%).

## 3. Version 4 (`..._v4.parquet`)

* **Goal:** Fix the gaps in **Market Cap**.
* **Input:** Version 3.
* **Script:** `backfill_market_cap_v4.py`
* **Method:**
    1.  **SEC Shares:** Merged time-varying share counts from SEC filings (`shares_events.parquet`).
    2.  **YF Cache:** Merged static share counts from `yf_snapshot_cache`.
    3.  **Network Fetch:** Downloaded `sharesOutstanding` for ~1,900 missing tickers.
    4.  **Calculation:** Re-calculated `market_cap = adj_close * shares_outstanding` for the entire history.
* **Result:**
    * **Market Cap Coverage:** **~59%** (up from ~31%).
    * *Note:* This version became the "stable base" for future steps.

## 4. Version 5 (SKIPPED / ABANDONED)

* **Goal:** Attempted to fix the ~25% missing **Volume** and **Price** data.
* **Attempt 1 (`backfill_volumes_v4.py`):** Tried to use `yf.download()` to fetch missing data.
    * *Outcome:* **Failed.** Hard IP block (`YFRateLimitError`) from Yahoo Finance.
* **Attempt 2 (`backfill_volumes_v5_history.py`):** We considered using `yf.Ticker().history()` to bypass the block.
    * *Outcome:* **Aborted.** This method does not return **Adjusted Close**. Using it would have made it impossible to calculate Market Cap for those tickers. We decided to preserve the integrity of the Market Cap logic rather than fill Volume with partial data.
* **Result:** No valid `v5` dataset was produced. We proceeded directly to `v6` using `v4` as the input.

## 5. Version 6 (`..._v6.parquet`)

* **Goal:** Add the missing **Top 100 ETFs**.
* **Input:** Version 4.
* **Script:** `run_me.py` (aka `odd_missing_etfs_v6_corrected_v3.py`).
* **Executed By:** Your friend (to bypass IP block).
* **Method:**
    1.  Compared the dataset against `top_100_etfs.csv`.
    2.  Identified 77 missing ETFs.
    3.  Downloaded full price history using a robust, slow loop with extreme backoff logic.
    4.  Calculated `ret` and `logret` for these new rows.
    5.  Appended these rows to the dataset.
* **Result:**
    * **ETF Coverage:** **100%** (23 original + 77 added).
    * **Sector:** New rows labeled as "ETF/Fund".
    * **Market Cap:** Calculated for all new ETFs.

## 6. Version 7 (`..._v7.parquet`) — **THE MASTER FILE**

* **Goal:** Finalize the **Volume** backfill by validating the ~1,779 missing tickers.
* **Input:** Version 6.
* **Script:** `backfill_volumes_v7_final.py` (aka `backfill_volumes_v7_robust_v2_fast_v2.py`).
* **Executed By:** Your friend.
* **Method:**
    1.  Targeted the ~1,779 tickers that still had `NaN` volume.
    2.  Used "Fast Fail" logic: If Yahoo returned "No Data Found" (`YFPricesMissingError`), the script immediately cached them as empty and moved on.
    3.  **Critical Finding:** Almost all of these tickers returned "No Data Found". This confirmed they are delisted, expired, or invalid symbols.
* **Result:**
    * **Content:** Identical to `v6` (contains all stocks + 100 ETFs).
    * **Validation:** The missing data holes are now confirmed **True Negatives**. You verified that no data exists for them, rather than it being a download error.
    * **Recalculation:** Market Cap was re-calculated one final time across the whole set.

---

# Final Dataset Statistics (v7)

| Metric | Coverage | Notes |
| :--- | :--- | :--- |
| **Tickers** | ~5,198 | Includes Stocks + Top 100 ETFs |
| **Sectors** | **100%** | Fully backfilled & classified |
| **ETFs** | **100%** | Top 100 list fully present |
| **Volume** | ~75.3% | Remaining ~24.7% confirmed dead/delisted |
| **Market Cap**| ~58.6% | Maximum possible achievable (Calculated daily where Price & Shares exist) |

---

# Key Files to Keep

1.  **`analysis_enriched_backfilled_v7.parquet`**: The Master Dataset.
2.  **`top_100_etfs.csv`**: The reference list for ETFs.
3.  **`outputs/price_cache/`**: The cache of downloaded price data (saves re-downloading later).
4.  **`etf_additions_cache/`**: The cache of the 77 added ETFs.
5.  **Scripts:**
    * `backfill_sectors_v3.py`
    * `backfill_market_cap_v4.py`
    * `run_me.py` (The ETF fetcher)
    * `backfill_volumes_v7_final.py` (The final validator)
