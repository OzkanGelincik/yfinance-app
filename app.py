"""
app.py — Event Study Studio (pandas-only)
-----------------------------------------
Run with:
    cd /path/to/your/project
    shiny run --reload app.py

What this app does (high level):
1) Loads your enriched Parquet dataset into memory (fast, typed).
2) Normalizes key columns (date, ticker), and builds a per-ticker trading-day
   index (tidx = 0,1,2,...) so we can make ±k-day windows ignoring weekends/holidays.
3) Constructs a unified 'form' column that represents an "event type"
   (either SEC filing forms like 10-Q, or synthetic labels 'SPLIT' / 'REVERSE_SPLIT').
4) Lets you filter events by form, date range, and sector; builds event windows;
   then shows summary stats and plots Cumulative Abnormal Returns (CAR) with a
   simple 95% CI ribbon. You can also download the raw window rows as CSV.
"""

from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
from shiny import App, ui, render, reactive
from shiny.ui import tags
from shinywidgets import output_widget, render_widget   # <-- add this
# (In the UI we’ll use ui.output_widget(...))
import plotly.graph_objects as go
import plotly.express as px

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Locate & load dataset                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Folder where this app.py lives (robust to "current working directory" issues) 
APP_DIR = Path(__file__).resolve().parent

OUT_DIR = APP_DIR / "outputs"

SEC_CSV = OUT_DIR / "sec_filing_descriptions.csv"

ETF_CSV = OUT_DIR / "top_100_etfs_described.csv"

www_dir = Path(__file__).parent / "www"

# Accept any of these filenames; first one found wins.
CANDIDATES = [
    "sample_backfilled_v7_1y_9col.parquet", # allow a tiny demo file
    "analysis_enriched_backfilled_v7.parquet",
]

# Find the first candidate that exists under outputs/
DATA_PQ = None
for name in CANDIDATES:
    p = APP_DIR / "outputs" / name
    if p.exists():
        DATA_PQ = p
        break
if DATA_PQ is None:
    raise FileNotFoundError(
        f"Could not find any of {CANDIDATES} in {APP_DIR/'outputs'}"
    )

# 1) DEFINE ONLY THE COLUMNS WE NEED
# This prevents loading 'open', 'high', 'low', 'volume', 'cik', etc. into RAM
cols_to_keep = [
    "date", "ticker", "close", "logret", "sector", 
    "filing_form", "is_split_day", "is_reverse_split_day"
]

# 2) LOAD DATA WITH OPTIMIZATIONS
AE = pd.read_parquet(DATA_PQ, columns=cols_to_keep)

# print(AE.sample(50))

# 3) OPTIMIZE MEMORY USAGE IMMEDIATELY
# Convert 'ticker' and 'sector' to category.
# (Strings take huge RAM; Categories map strings to tiny integers)
AE["ticker"] = AE["ticker"].astype("category")
AE["sector"] = AE["sector"].astype("category")

# Downcast floats to float32 (halves memory usage for numbers)
f_cols = AE.select_dtypes(include=['float64']).columns
AE[f_cols] = AE[f_cols].astype("float32")

# ── Normalize core columns (prevents a lot of downstream headaches) ─────────
# 1) date → pandas datetime (NaT on bad values), drop timezone, keep only the day
AE["date"] = (
    pd.to_datetime(AE["date"], errors="coerce")
      .dt.tz_localize(None)
      .dt.normalize()
)

# 2) ticker → Ensure consistency (Category handles this better, but if you need uppercase string logic:)
# Note: Since we used category above, we verify they are upper. 
# If your parquet tickers are already uppercase, you can skip this.
# If you must force upper, do it before converting to category to save RAM:
# (Here we assume they are reasonably clean, or we process them as categories)

# 2) ticker → uppercase strings (consistent joins/filters)
# AE["ticker"] = AE["ticker"].astype(str).str.upper()

# 3) Sort (Sorting is expensive! doing it after dropping columns is faster)
AE = AE.sort_values(["ticker", "date"]).reset_index(drop=True)

# 4) Trading-day index per ticker:
#    For each ticker, 0,1,2,... in chronological order (ignores real calendar gaps)
AE["tidx"] = AE.groupby("ticker").cumcount()

# 5) Ensure log returns exist (if not, compute from 'close')
if "logret" not in AE.columns and "close" in AE.columns:
    AE["close"] = pd.to_numeric(AE["close"], errors="coerce")
    # logret_t = ln(close_t / close_{t-1}) calculated per ticker
    AE["logret"] = (
        AE.groupby("ticker")["close"]
          .apply(lambda s: np.log(s / s.shift(1)))
          .reset_index(level=0, drop=True)
          .astype("float32") # Keep it small
    )

# NOTE: Deprecated helper. Event Study now uses the long EVENTS table
# (build_events_long) and no longer relies on AE["form"].
# If you need quick inline tagging for ad-hoc inspection, you can re-enable this.
# AE["form"] = ...

# ── Build a unified 'form' column: filing form OR split label ───────────────
# Use object dtype to avoid NumPy "dtype promotion" issues when mixing strings/NaN
# AE["form"] = (
#     AE["filing_form"].astype("object")
#     if "filing_form" in AE.columns
#     else pd.Series([None] * len(AE), index=AE.index, dtype="object")
# )

# Safe boolean masks even if columns are missing; fillna(False) to avoid NaNs in logic
# rev = (
#     AE["is_reverse_split_day"].fillna(False)
#     if "is_reverse_split_day" in AE.columns
#     else pd.Series(False, index=AE.index)
# )
# fwd = (
#     AE["is_split_day"].fillna(False)
#     if "is_split_day" in AE.columns
#     else pd.Series(False, index=AE.index)
# )

# Only fill split labels where no filing form exists (don’t overwrite true forms)
# AE.loc[AE["form"].isna() & rev, "form"] = "REVERSE_SPLIT"
# AE.loc[AE["form"].isna() & fwd, "form"] = "SPLIT"

# ---- Helpers for returns & shapes ----
def _simple_returns(logret: pd.Series) -> pd.Series:
    """Convert log returns to simple returns; fill NaNs with 0 (flat)."""
    return np.expm1(logret.fillna(0.0))

def _dater_default(df: pd.DataFrame, days_back: int = 252):
    """Default date range: last ~1Y of data (trading days)."""
    hi = pd.to_datetime(df["date"].max())
    lo = max(pd.to_datetime(df["date"].min()), hi - pd.Timedelta(days=days_back))
    return str(lo.date()), str(hi.date())
    
# ── Build a proper long "EVENTS" table (one row per distinct event) ─────────    
def build_events_long(AE: pd.DataFrame) -> pd.DataFrame:
    parts = []

    # Filing events
    if "filing_form" in AE.columns:
        f = AE.loc[AE["filing_form"].notna(), ["ticker", "date", "filing_form"]].copy()
        f = f.rename(columns={"filing_form": "event_type"})
        parts.append(f)

    # Split events
    if "is_split_day" in AE.columns:
        s = AE.loc[AE["is_split_day"] == True, ["ticker", "date"]].copy()
        s["event_type"] = "SPLIT"
        parts.append(s)

    if "is_reverse_split_day" in AE.columns:
        r = AE.loc[AE["is_reverse_split_day"] == True, ["ticker", "date"]].copy()
        r["event_type"] = "REVERSE_SPLIT"
        parts.append(r)

    if not parts:
        ev = pd.DataFrame(columns=["ticker", "date", "event_type"])
    else:
        ev = pd.concat(parts, ignore_index=True).drop_duplicates()

    # Attach sector if present (use a minimal deduped slice)
    if "sector" in AE.columns:
        ev = ev.merge(
            AE.loc[:, ["ticker", "date", "sector"]].drop_duplicates(),
            on=["ticker", "date"],
            how="left",
        )

    # Flag same-day overlaps per ticker
    counts = ev.groupby(["ticker", "date"], observed=True).size().rename("n_events")
    ev = ev.merge(counts, on=["ticker", "date"], how="left")
    ev["is_overlap"] = ev["n_events"] > 1

    # Ensure proper dtypes
    ev["date"] = pd.to_datetime(ev["date"]).dt.normalize()
    ev["ticker"] = ev["ticker"].astype(str).str.upper()
    return ev

EVENTS = build_events_long(AE)

# ── Choices for the UI ──────────────────────────────────────────────────────
# ── Choices for the UI (as you already had) ──
# Replace empty strings (and pure-whitespace strings) with NaN
AE["sector"] = AE["sector"].replace("", np.nan)
AE["sector"] = AE["sector"].where(AE["sector"].astype(str).str.strip() != "", np.nan)

sector_choices = (
    sorted(AE["sector"].dropna().unique().tolist())
    if "sector" in AE.columns else []
)

event_types = sorted(EVENTS["event_type"].dropna().unique().tolist())

# This "forms" is depricated as I built a more comprehensive approach to studying the filings, split, and reverse split events. 
# Ignore this "forms" unless you activate the previous approach that's commented out above.
# forms = [
#     "10-Q", "10-K", "8-K", "S-1", "S-1/A", "DRS", "DRS/A", "UPLOAD",
#     "SC 13G", "SC 13G/A", "SPLIT", "REVERSE_SPLIT",
# ]
ticker_choices = sorted(AE["ticker"].dropna().unique().tolist())
dlo, dhi = _dater_default(AE, 252)

# Add a tiny color-map helper (top of file, near other helpers)
def _build_color_map(labels):
    """Stable ticker→color map (same color for a ticker across all pies)."""
    palette = (px.colors.qualitative.D3
               + px.colors.qualitative.Set2
               + px.colors.qualitative.Plotly)
    labs = sorted(pd.Index(labels).unique())
    return {lab: palette[i % len(palette)] for i, lab in enumerate(labs)}

# Update your pie factory to accept the map and enforce order
def _pie_fig(*, names, values, title, hole=0.35, unit=None, percent=True, color_map=None):
    # if a color_map is given, reindex to that order to keep legend/slice order stable
    ser = pd.Series(values, index=pd.Index(names, name="ticker"))
    if color_map:
        ordered = [t for t in color_map.keys() if t in ser.index]
        ser = ser.reindex(ordered)

    fig = px.pie(
        names=ser.index,
        values=ser.values,
        hole=hole,
        title=title,
        color=ser.index,                       # color by ticker
        color_discrete_map=color_map or {},    # enforce our mapping
    )

    # hover text
    if unit == "currency":
        ht = "%{label}: $%{value:,.0f}"
    elif unit == "shares":
        ht = "%{label}: %{value:,.4f} sh"
    else:
        ht = "%{label}: %{value}"
    if percent:
        ht += " (%{percent})"
    fig.update_traces(hovertemplate=ht + "<extra></extra>", textinfo="percent")

    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), title_x=0.5, legend_title_text="Ticker")
    return fig


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  UI Layout                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# page_fluid: responsive page
# layout_sidebar: inputs on left, outputs on right
# Each ui.input_* creates a reactive input accessible via `input.<id>()` in server.

head_links = ui.head_content(
    # (Optional) helper to jump to a tab by label text
    ui.tags.script("""
      window.goToTabByText = function(txt){
        var links = document.querySelectorAll('.nav-link');
        for (const a of links) {
          if ((a.textContent || a.innerText).trim() === txt) { a.click(); return true; }
        }
        return false;
      };
    """),

    # Icons (keep if used)
    ui.tags.link(rel="stylesheet",
      href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css",
      crossorigin="anonymous", referrerpolicy="no-referrer"),
    ui.tags.link(rel="stylesheet",
      href="https://cdnjs.cloudflare.com/ajax/libs/academicons/1.9.4/css/academicons.min.css",
      crossorigin="anonymous", referrerpolicy="no-referrer"),

    # Global styles
    ui.tags.style(
        """          
        .about-badges .badge { font-weight:600; padding:.45rem .75rem; }
        .about-badges .pill-nycdsa { background:#ffb3ba; color:#000; }
        .about-badges .pill-cornell { background:#ffffba; color:#000; }
        .about-badges .pill-ops { background:#b3ebf2; color:#000; }

        .btn-icon { display:inline-flex; align-items:center; justify-content:center;
                  width:40px; height:40px; padding:0; border-radius:.5rem; }
        .btn-icon i { line-height:1; font-size:1.15rem; }

        .about-left  { padding-right:1rem; }
        .about-right { padding-left:1rem; }
        @media (min-width: 992px) {
        .about-left  { padding-right:1.25rem; border-right:1px solid #eee; }
        .about-right { padding-left:1.25rem; }
        }

        /* Appendix/plain-math helpers */
        .nowrap { white-space: nowrap; }

        /* Optional video styles */
        .about-video{
        width:50%; height:auto; display:block;
        margin:.5rem 0 1.25rem; border:1px solid #e9ecef;
        border-radius:.75rem; box-shadow:0 1px 2px rgba(0,0,0,.06);
        }
        .video-caption{ margin-top:-.5rem; margin-bottom:1rem; color:#6c757d; font-size:.9rem; }
                  
        /* Padding prevents first-child margin collapse */
        .about, .appendix { padding-top: .25rem; }

        /* Add vertical breathing room before headings */
        .about h2, .about h3, .about h4, .about h5, .about h6,
        .appendix h2, .appendix h3, .appendix h4, .appendix h5, .appendix h6 {
        margin-top: 1.25rem !important;   /* adjust as you like */
        margin-bottom: 0.5rem !important;
        }

        /* Don’t over-space the very first heading inside each panel */
        .about h2:first-of-type, .about h3:first-of-type, .about h4:first-of-type, .about h5:first-of-type, .about h6:first-of-type,
        .appendix h2:first-of-type, .appendix h3:first-of-type, .appendix h4:first-of-type, .appendix h5:first-of-type, .appendix h6:first-of-type {
        margin-top: 0.25rem !important;
        }
    """),

    # … your <link> tags …
    ui.tags.style("""
      :root { --block-gap: 1.25rem; }  /* tweak once, affects all outputs */

      /* Most Shiny outputs */
      .shiny-output,
      .shiny-plot-output,
      .shiny-text-output,
      .shiny-table-output,
      .shiny-html-output,
      .shiny-data-frame-output,
      /* Plotly/HTML-widget containers */
      .html-widget,
      /* shinywidgets output container */
      .py-shiny-output-widget,
      /* DataFrame grid (shiny.render.data_frame) */
      .data-grid,
      .dataframe {
        margin-bottom: var(--block-gap);
      }

      /* Optional: slightly tighter spacing inside sidebars for inputs */
      .sidebar .shiny-input-container { margin-bottom: .75rem; }
    """),

)


# Define the markdown content for "About" as a string
about_markdown = r"""
## **What this app does**
**1)** **Portfolio Simulator — build & backtest portfolios**

Simulates a buy-and-hold portfolio from your chosen tickers with equal-weight (default) or inverse-price starting weights; customizable date range and starting cash.

<div>
  <video class="about-video" controls preload="metadata" playsinline poster="portfolio_simulator_poster.png">
    <source src="https://yfinance-app-videos.s3.amazonaws.com/portfolio_simulator_demo.mp4" type="video/mp4">
    Your browser doesn’t support HTML5 video. You can
    <a href="https://yfinance-app-videos.s3.amazonaws.com/portfolio_simulator_demo.mp4" download>download the demo</a>.
  </video>
  <div class="video-caption">Quick tour of the Portfolio Simulator workflow.</div>
</div>

**2)** **Sector Explorer — build indexed sector lines**

Builds equal-weight (per day) sector return series from constituents and cumulates them to indices that start at 1.

<div>
  <video class="about-video" controls preload="metadata" playsinline poster="sector_explorer_poster.png">
    <source src="https://yfinance-app-videos.s3.amazonaws.com/sector_explorer_demo.mp4" type="video/mp4">
    Your browser doesn’t support HTML5 video. You can
    <a href="https://yfinance-app-videos.s3.amazonaws.com/sector_explorer_demo.mp4" download>download the demo</a>.
  </video>
  <div class="video-caption">From daily sector averages to cumulative indices.</div>
</div>

**3)** **Event Study — average (cumulative) log returns around events**

Select filing/split types, a window (±k trading days), and optional sector filters; view average CAR with a 95% CI.

<div>
  <video class="about-video" controls preload="metadata" playsinline poster="event_study_poster.png">
    <source src="https://yfinance-app-videos.s3.amazonaws.com/event_study_demo.mp4" type="video/mp4">
    Your browser doesn’t support HTML5 video. You can
    <a href="https://yfinance-app-videos.s3.amazonaws.com/event_study_demo.mp4" download>download the demo</a>.
  </video>
  <div class="video-caption">Event selection → windows → CAR chart & export.</div>
</div>


<span>
  <a href="#tab-Appendix"
     class="link-button"
     onclick="if(!window.goToTabByText||!goToTabByText('Appendix')){return true;} return false;">
     Appendix: Learn more about what each function does and how returns were calculated (i.e., simple and log returns) →
  </a>
</span>
<div class="mt-3"></div>

## **Tech stack** - **Python Shiny (UI)**
  - pandas, NumPy (wrangling & returns)  
  - Plotly (interactive charts)
- **Data sources** - Yahoo Finance via 
    - `yfinance` (OHLCV, adj close)  
    - `yahoo_fin` (NASDAQ/NYSE ticker lists)
    - `yahooquery` (market cap, sector, industry)  
  - SEC EDGAR API (submissions + companyfacts JSON (`/api/xbrl/companyfacts/CIK{cik}.json`) via `requests`) for CIK/SIC, filings, shares/public float
- **Caching/IO** - Parquet for fast reloads; local JSON cache for SEC responses; CSV export from the app. *No DuckDB used.*

## **Data & caveats** 
- Educational use only — **not financial advice**.  
- Market data may be delayed and subject to API limits.  
- Results are demo estimates; verify independently.

## **Contact** 
Use the buttons above (LinkedIn, GitHub, ResearchGate, Google Scholar).  

## **Acknowledgments**
*Grateful to the NYC Data Science Academy community—especially Vinod Chugani, Jonathan Harris, Joe Lu, David Wasserman, Andrew Dodd, and Luke Lin—for thoughtful, actionable feedback. Thanks also to the open-source maintainers and data providers—`yfinance`, `yahoo_fin`, `yahooquery`, and SEC EDGAR—that made this project possible.*
"""

# Define the markdown content for "Dataset Build" as a string
dataset_build_markdown = """
## **Dataset Construction Flow**

A top-to-bottom look at how the `analysis_enriched` dataset was built, from raw data harvesting to final assembly.

---

#### **Step 1: Harvest Ticker Universe**
The process begins by fetching a comprehensive list of all stock tickers listed on the NASDAQ and NYSE exchanges.
* **Package:** `yahoo_fin`
* **Code:** `si.tickers_nasdaq()`, `si.tickers_nyse()`

#### **Step 2: Download Daily Price History**
With the list of tickers, we fetch 3+ years of daily Open, High, Low, Close, and Volume (OHLCV) data for every stock.
* **Package:** `yfinance`
* **Outputs:** `open`, `high`, `low`, `close`, `adj_close`, `volume`
* **Code:** `yf.download(tickers, period="3y", interval="1d")`

#### **Step 3: Compute Daily Returns**
Using the `adj_close` column (which accounts for dividends and splits), we calculate the simple (`ret`) and logarithmic (`logret`) daily returns.
* **Package:** `pandas`
* **Outputs:** `ret`, `logret`
* **Code:**
    ```python
    df.groupby('ticker')['adj_close'].pct_change()
    np.log(df['adj_close'] / df.groupby('ticker')['adj_close'].shift(1))
    ```

#### **Step 4: Harvest Static Metadata**
We gather static company info from two sources: company identifiers (CIK, SIC) from the SEC and business classifications (sector, industry) from Yahoo.
* **Packages:** `requests`, `yahooquery`
* **Outputs:** `cik`, `sic`, `sic_desc`, `sector`, `industry`, `market_cap`
* **Code:**
    ```python
    requests.get("[https://data.sec.gov/submissions/CIK](https://data.sec.gov/submissions/CIK)...")
    T.get_modules('price,assetProfile')
    ```

#### **Step 5: Harvest Point-in-Time Events**
Next, we collect data that only appears on specific dates: stock splits from Yahoo and time-varying `shares_outstanding` and `EntityPublicFloat` (in USD) from the SEC's CompanyFacts API.
* **Packages:** `yfinance`, `requests`
* **Code:**
    ```python
    yf.Ticker(t).get_actions()
    requests.get(".../api/xbrl/companyfacts/CIK...")
    ```

#### **Step 6: Backfill Share Data Gaps**
The SEC API doesn't cover all tickers (e.g., some foreign listings). For any tickers still missing share data, we use `yfinance` as a fallback to get the latest `sharesOutstanding` and `floatShares` snapshot.
* **Package:** `yfinance`
* **Code:**
    ```python
    yf.Ticker(t).info['floatShares']
    yf.Ticker(t).fast_info['shares']
    ```

#### **Step 7: Assemble, Fill, and Annotate**
This is the most critical step. All data sources are merged into the daily price panel.
* **Forward-Fill:** We carry the last known `shares_outstanding` and `float_shares` value forward in time.
* **Split Annotation:** Split events are merged to create flags like `is_split_day` and `split_cum_factor`.
* **Filing Annotation:** A full filing history is merged to add `filing_form` and `days_since_filing` to *every single row*.
* **Package:** `pandas`
* **Outputs:** `shares_outstanding`, `float_shares`, `split_ratio`, `filing_form`, etc.
* **Code:**
    ```python
    df.groupby('ticker')[col_to_fill].ffill()
    pd.merge_asof(daily_panel, filings, direction='backward', ...)
    ```

#### **Step 8: Save Meta Dataset (v4 - Market Cap Fixed)**
The final, fully assembled DataFrame is saved. This version (`v4`) successfully backfilled `sector` to 100% and `market_cap` to ~60% by calculating it daily (`adj_close * shares_outstanding`).
* **Note on v5:** An attempt was made to backfill missing `volume` data using `yf.download()`, but it was skipped due to IP rate-limiting blocks. We proceeded to v6 instead.
* **Meta Files:** `analysis_enriched_backfilled_v4.parquet`
* **Meta Data Content:** 31 columns (see dictionary) for 5,120+ tickers.

#### **Step 9: Step 9: Add Missing ETFs (v6) & Validate Volumes (v7)**
We identified that 77 of the top 100 ETFs were missing. We successfully downloaded their full history and appended them (`v6`). We then ran a validation pass on the ~1,779 tickers with missing volumes, confirming they are delisted/dead (`v7`).
* **Final Master File:** `analysis_enriched_backfilled_v7.parquet`
* **Key Stats:** 100% Sector Coverage, 100% Top ETF Coverage, Validated "True Negative" missing data.

#### **Step 10: Subset Data for Shiny App**
The final dataset used in this app is a lightweight subset of the meta file `v7`. It is a 1.2M+ row DataFrame saved as `sample.parquet` with only the columns needed for the app: `date`, `ticker`, `close`, `logret`, `filing_form`, `sector`, `is_split_day`, `is_reverse_split_day`, and `market_cap`. For practical reasons, only the last year of data (approx. 2024-09-30 to 2025-09-30) was kept.
* **Final File:** `sample.parquet`

---

#### **Meta Data Dictionary**

| Variable | dtype | Description |
| :--- | :--- | :--- |
| `date` | `datetime64[ns]` | Trading date, normalized to midnight (YYYY-MM-DD), no timezone. One row per ticker per trading day. |
| `ticker` | `object` (`string`) | Stock symbol (uppercased). Primary identifier along with date. |
| `open` | `float` | Opening price for the trading day. |
| `high` | `float` | Highest price traded during the day. |
| `low` | `float` | Lowest price traded during the day. |
| `close` | `float` | Closing price for the day. Main price used for returns. |
| `adj_close` | `float` | Adjusted close price (accounts for splits and sometimes dividends), as provided by yfinance. |
| `volume` | `float` / `int` | Number of shares traded on that day. |
| `ret` | `float` | Simple daily return: roughly close_t / close_{t-1} - 1. Intuitive: 0.05 = +5%. |
| `logret` | `float` | Log daily return: log(close_t / close_{t-1}). Adds nicely over time; used in event study & app. |
| `market_cap` | `float` | Approximate market capitalization on that date (close * shares_outstanding), if available. |
| `sector` | `object` | Sector classification (e.g., Healthcare, Financial Services). Often missing for microcaps / obscure tickers. |
| `industry` | `object` | More granular industry classification, if available. |
| `cik` | `object` / `int`-like | SEC CIK (Central Index Key) for the issuer, used to link filings to the company. |
| `sic` | `object` / `int`-like | Standard Industrial Classification code (numeric). |
| `sic_desc` | `object` | Text description of the SIC code (e.g., "HOSPITALS", "CRUDE PETROLEUM & NATURAL GAS"). |
| `recent_form` | `object` | Most recent filing form (e.g., 10-K, 10-Q) as a snapshot from SEC submissions API. |
| `recent_filing_date` | `datetime64[ns]` | Date of the most recent filing (snapshot, not time-varying). |
| `shares_outstanding` | `float` | Number of shares outstanding as of this date. Used with close to compute market cap. |
| `float_shares` | `float` | Float shares (freely tradable shares, excluding locked-up or tightly held ones), if available. |
| `free_float` | `float` | Free float ratio, typically float_shares / shares_outstanding. Between 0 and 1 (e.g. 0.35 = 35% free float). |
| `filing_form` | `object` | Filing form on this exact date for this ticker (e.g. 10-Q, 8-K, S-1, SC 13G/A). NaN on non-filing days. This is your "event-day" form. |
| `last_filing_date` | `datetime64[ns]` | Date of the latest filing on or before this date. Similar to `recent_filing_date`; exact logic may differ slightly in the notebook. |
| `is_filing_day` | `bool` | True if a filing occurs on this date for this ticker (i.e. there is a `filing_form` that day). |
| `days_since_filing` | `int` | Number of calendar days since the last filing (`date - last_filing_date` in days). If no prior filing exists, often NaN or a sentinel. |
| `split_ratio` | `float` | Split ratio if there's a split event on this date (e.g. 2.0 = 2-for-1, 0.2 = 1-for-5 reverse split). |
| `is_split_day` | `bool` | True on forward split dates (share count increases, price drops). |
| `is_reverse_split_day` | `bool` | True on reverse split dates (share count decreases, price jumps). |
| `split_cum_factor` | `float` | Cumulative split adjustment factor up to this date. Starts at 1.0, multiplies by each split_ratio. |
"""

appendix_markdown = r"""
## **About this app**  
An interactive toolkit with three workflows—Portfolio Simulator (buy-and-hold with Equal or Inverse-price weights), Sector Explorer (indexed sector lines with equal-weight members and rolling stats), and Event Study (AR/CAR with expected return = 0). It highlights how choices like weight scheme, date range, and event window change results. *No rebalancing; no baseline model.*

**1) Portfolio Simulator — build & backtest portfolios**
- **What it does:** Simulates a buy-and-hold portfolio from your tickers with two weight options (Equal or Inverse-price), custom date range, and starting cash.
- **How to use:** 
  1) Enter tickers separated by spaces (no commas), e.g., AAPL, MSFT, NVDA
  2) Set dates
  3) Choose weighting
  4) Click "Simulate".
- **Weights** (2-stock example)
  - Equal weight: <span class="nowrap">w<sub>i</sub> = 1/N</span>. With 10,000 dollars, AAPL 200 dollars/share &amp; MSFT 100 dollars/share → 5,000 dollars each → 25 AAPL, 50 MSFT.
  - Inverse-price: <span class="nowrap">w<sub>i</sub> = (1/P<sub>i</sub>) / Σ<sub>j</sub>(1/P<sub>j</sub>)</span>. Same prices → ≈33% AAPL / 67% MSFT → 3,333 AAPL (≈16.7 shares), 6,667 MSFT (≈66.7 shares).
- **Outputs:** Portfolio wealth (\$) time series, total period return (%), optional daily return series, start→end wealth summary, and per-ticker normalized cumulative lines; CSV export. *(No turnover or trade list—there’s no rebalancing.)*
- **Under the hood:** Weights are fixed once at <span class="nowrap">t<sub>0</sub></span> (equal or inverse-price using the first available `close`). Returns use simple returns derived from adj-close log returns; portfolio wealth is a fixed linear combo of each asset’s growth index. *No rebalancing*, *no transaction costs*; *fractional shares implied*.

**2) Sector Explorer — compare trends across sectors**
- **What it does:** Benchmarks selected sectors/industries over a window with indexed performance (base=1), rolling returns, and a return-correlation view.
- **How to use:** 
  1) Pick the date range and sectors
  2) Toggle "Equal-weight within sector?"  
  3) Choose a rolling window (e.g., 20/60 days)
  4) Click "Build sector indices" (index=1 at start)
- **Outputs:** Performance time series (indexed), a summary table of individual sector returns.
- **Beta note — “Equal-weight within sector”**
  - This selector is in *beta*; *currently it has no effect*. Sectors are always computed as an equal-weight average of their member tickers (not cap-weighted).
    - **What “equal-weight within sector” means (example)**
      - Tech = {AAPL, MSFT} → 50%/50%; sector daily return = mean(<span class="nowrap">r<sub>AAPL</sub></span>, <span class="nowrap">r<sub>MSFT</sub></span>).
      - Health = {JNJ, PFE, MRK} → 33% each; sector daily return = mean(<span class="nowrap">r<sub>JNJ</sub></span>, <span class="nowrap">r<sub>PFE</sub></span>, <span class="nowrap">r<sub>MRK</sub></span>).  
      - Each sector’s line is then indexed to 1 at the start of your window.

**3) Event Study — measure market reaction around news**
- **What it does:** Aligns price moves around one or more event dates (e.g., earnings, SEC filings) to compute abnormal return (AR) and cumulative abnormal return (CAR) with expected return = 0 (so <span class="nowrap">AR<sub>t</sub> = r<sub>t</sub>; CAR = Σ AR<sub>τ</sub></span>).
- **How to use:**
  1) Choose a ticker and an event window (e.g., −10 to +10 trading days).
  2) Enter one or more dates (YYYY-MM-DD; spaces or commas are fine).
  3) Click "Run".
- **Outputs:** Per-event AR/CAR tables; average AR/CAR across events when multiple dates are provided; a day-0-aligned chart; CSV export.
- **Under the hood:** Uses simple returns derived from adj-close log returns.
  - If an event date falls on a non-trading day, day 0 = the next trading day.
  - Averages across multiple events are equal-weight.
  - *No transaction costs; educational use only.*
> *Future idea:* add optional baselines (constant-mean or market model vs. a benchmark) and an explicit estimation window.

---

## **Returns: definitions & conversions**

- **Definitions**
    - **Simple return:** <span class="nowrap">r = (P<sub>t</sub> − P<sub>t−1</sub>) / P<sub>t−1</sub> = P<sub>t</sub> / P<sub>t−1</sub> − 1</span>
    - **Log return:** <span class="nowrap">g = ln(P<sub>t</sub> / P<sub>t−1</sub>) = ln(1 + r)</span>
<br><br>
- **Convert between them**
    - <span class="nowrap">simple → log: g = ln(1 + r)</span>
    - <span class="nowrap">log → simple: r = e<sup>g</sup> − 1</span> (this is what <code>np.expm1(g)</code> does)
<br><br>
- **Over multiple days**
    - **Cumulative simple:** <span class="nowrap">R = ∏<sub>i</sub> (1 + r<sub>i</sub>) − 1</span>
    - **Cumulative log:** <span class="nowrap">G = Σ<sub>i</sub> g<sub>i</sub></span>
    - **Relationship:** <span class="nowrap">G = ln(1 + R)</span> and <span class="nowrap">R = e<sup>G</sup> − 1</span>
<br><br>
- **Rule of thumb:** for small returns, <span class="nowrap">ln(1 + r) ≈ r − r<sup>2</sup>/2</span> (so log ≈ simple).

**Example.** If <span class="nowrap">r = 0.05</span>, then <span class="nowrap">g = ln(1.05) ≈ 0.04879</span>. Conversely, <span class="nowrap">g = 0.04879 → r = e<sup>0.04879</sup> − 1 ≈ 0.05</span>.

**Why `np.expm1(CAR)`?** CAR is a **sum of log returns**, so <span class="nowrap">percent ≈ e<sup>CAR</sup> − 1</span>, which is exactly what <code>np.expm1</code> computes.
"""

etfs_markdown = '''

# **Understanding the top-100 ETFs table**
These listed ETFs were taken from [ETF Database](https://etfdb.com/compare/market-cap/) on 24/11/2025.

---

## **Tracking_Index**  
  The specific benchmark index the ETF is designed to track (e.g., S&P 500 Index, CRSP US Total Market Index, Nasdaq-100 Index).  
  For active funds, this is noted as `"N/A (Active)"`.

---
  
## **Methodology**  
How the ETF tracks its index:

  - **Full Replication**  
    The fund buys all the securities in the index with the same weightings.  
    *(Common for S&P 500 and Nasdaq-100 ETFs).*

  - **Sampling**  
    The fund buys a representative sample of securities to mimic the index's risk/return profile without holding every single stock/bond.  
    *(Common for Total Market and Bond ETFs).*

  - **Active Management**  
    The fund manager makes active investment decisions rather than tracking an index (e.g., JEPI, JEPQ).

  - **Physical Backing**  
    The fund holds the actual asset (e.g., gold or bitcoin).

---
    
## **Diversification**  
  Indicates the breadth of the fund's holdings (e.g., broad market, sector-specific, or single commodity).

---

## **Exposure Description**

#### **1. General Format**

- **Equities (Stocks):**
`Region/Style: ~% Top Sector 1, ~% Top Sector 2...`
    - Example:  
    *"US Large Cap: ~31% Tech, ~14% Fin..."* means the fund invests in big US companies, and its performance is most heavily influenced by Technology (31%) and Financials (14%).

- **Bonds (Fixed Income):**  
`Credit Quality/Type: ~% Breakdown by Issuer Type`
    - Example:  
    *"US Inv Grade Bonds: ~43% Treasuries..."* means the fund holds safe US debt, split between Government bonds (Treacheries) and other types.

- **Commodities/Alternative:**  
`Type: 100% Underlying Asset`
    - Example:  
    *"Commodity: 100% Physical Gold"*

#### **2. Abbreviations Key**

- **Sectors (What industry the companies are in)**

    - **Tech:** Information Technology (e.g., Apple, Microsoft, Nvidia)  
    - **Fin:** Financials (e.g., Chase, Visa, Berkshire Hathaway)  
    - **Health:** Healthcare (e.g., UnitedHealth, Eli Lilly)  
    - **Cons Disc:** Consumer Discretionary (non-essential goods like Amazon, Tesla, McDonald's)  
    - **Comm Svcs:** Communication Services (e.g., Google/Alphabet, Meta, Netflix)  
    - **Ind:** Industrials (e.g., Caterpillar, GE, Aerospace)  
    - **Cons Staples:** Consumer Staples (essential goods like P&G, Walmart, Coke)  
    - **Real Estate / REITs:** Real Estate Investment Trusts  

- **Bond Types**

    - **Treasuries:** Debt issued by the US Government (safest).  
    - **MBS:** Mortgage-Backed Securities (pools of home loans, usually agency-backed).  
    - **Corporates:** Debt issued by companies.  
    - **Inv Grade:** Investment Grade (high credit quality, lower risk).  
    - **High Yield / Junk:** Non-investment grade (lower credit quality, higher risk/yield).  
    - **Muni:** Municipal Bonds (issued by states/cities, often tax-exempt).  

- **Regions

    - **Dev Markets:** Developed Markets (advanced economies like UK, Japan, France, Canada).  
    - **Emerging Markets:** Developing economies (China, India, Brazil, Taiwan).  

#### **3. Important Note on Percentages (`~`)**

You will see the tilde symbol (`~`) before percentage numbers. This indicates an **approximation**.

- ETF weightings change daily as stock prices move.  
- The values provided are based on typical benchmark index weightings (e.g., the S&P 500 is historically roughly 30% Tech).  
- `"100%"` implies the fund is a *pure-play* on that specific asset (e.g., a Gold trust or a specific sector fund).

---

## **Cost (Expense Ratio)**  
  Estimated expense tier:
  - Ultra Low: ~0.03%  
  - Low: ~0.10%  
  - Moderate: >0.15%

---
  
## **Trading Flexibility**  
  All top 100 ETFs have high liquidity; specific highly traded ones like SPY/QQQ are noted as **"Very High"**.

---  
  
## **Transparency**  
  Almost all ETFs disclose holdings daily (**"High"**).

---

## **Tax Efficiency**  
  - Equity ETFs: generally **High**  
  - Bond ETFs: produce taxable interest (**Moderate**)  
  - Gold/Commodities: often have special tax treatments (**Low**)

'''

# This is the main UI definition you can use in your nav_panel
dataset_build_ui = ui.div(
    ui.markdown(dataset_build_markdown),
    # Add some padding for better readability
    {"style": "padding: 2rem;"}
)

app_ui = ui.page_fluid(
    head_links,
    ui.h2("Portfolio, Sector & Event Lab", class_="fw-bold"),
    ui.p("Build buy-and-hold backtests, benchmark sectors, and quantify reactions to news."),
    ui.navset_tab(

        ui.nav_panel(
            "About",
            ui.div(
                ui.layout_columns(
                    ui.div(
                        ui.img(
                            src="new_profile_picture.png",
                            alt="Portrait of Ozkan Gelincik",
                            class_="mb-3", width="100%"
                        ),
                        ui.h4("About the author", class_="fw-bold"),
                        ui.markdown(
                            '''
                            I’m Ozkan Gelincik—cancer-research operations leader turned data scientist. After 10+ years at Weill Cornell Medicine advancing cancer-prevention research and publishing in top-tier medical journals, I now build end-to-end data pipelines, machine-learning models, and interactive visualizations that turn messy data into clear, actionable decisions.
                            '''
                        ),
                        ui.hr(),
                        ui.div(
                            ui.span("NYC Data Science Academy · DS/ML",
                                    class_="badge rounded-pill pill-nycdsa"),
                            ui.span("Weill Cornell Medicine · 10+ years",
                                    class_="badge rounded-pill pill-cornell"),
                            ui.span("Cancer Research Operations",
                                    class_="badge rounded-pill pill-ops"),
                            class_="about-badges d-flex flex-wrap gap-2 mb-4"
                        ),
                        ui.hr(),
                        ui.div(
                            ui.a(
                                ui.tags.i(class_="fa-brands fa-linkedin-in fa-lg"),
                                href="https://www.linkedin.com/in/ozkangelincik/",
                                class_="btn btn-outline-primary btn-sm me-2 btn-icon",
                                target="_blank", rel="noopener noreferrer",
                                **{"aria-label": "LinkedIn"},
                            ),
                            ui.a(
                                ui.tags.i(class_="fa-brands fa-github fa-lg"),
                                href="https://github.com/OzkanGelincik",
                                class_="btn btn-outline-secondary btn-sm me-2 btn-icon",
                                target="_blank", rel="noopener noreferrer",
                                **{"aria-label": "GitHub"},
                            ),
                            ui.a(
                                ui.tags.i(class_="fa-brands fa-researchgate fa-lg"),
                                href="https://www.researchgate.net/profile/Ozkan-Gelincik",
                                class_="btn btn-outline-dark btn-sm me-2 btn-icon",
                                target="_blank", rel="noopener noreferrer",
                                **{"aria-label": "ResearchGate"},
                            ),
                            ui.a(
                                ui.tags.i(class_="ai ai-google-scholar-square ai-lg"),  # square variant is nice
                                href="https://scholar.google.com/citations?user=2bcmUHoAAAAJ&hl=en",
                                class_="btn btn-outline-success btn-sm btn-icon",
                                target="_blank", rel="noopener noreferrer",
                                **{"aria-label": "Google Scholar"},
                            ),
                        ),
                        class_="about-left",
                    ),
                    ui.div(
                        ui.markdown(about_markdown),
                        class_="about-right",
                    ),
                    col_widths=(3, 8),
                    row_classes="gx-0",  # let our CSS control the gap
                ),
                class_="about",
            ),
        ),


        ui.nav_panel("Dataset Build", 
            ui.div(     
                ui.markdown(dataset_build_markdown),
                # Add some padding for better readability
                {"style": "padding: 2rem;"}
            ),
        ),


        # 2) PORTFOLIO SIMULATOR
        ui.nav_panel(
            "Portfolio Simulator",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_selectize("p_tickers", "Pick tickers (1–10)", ticker_choices, multiple=True),
                    ui.input_numeric("p_cash", "Initial cash ($)", value=10_000, min=100),
                    ui.input_checkbox("p_equal", "Equal-weight portfolio?", value=True),
                    ui.input_date_range("p_dater", "Backtest range (Data available: 9/30/2024-9/30/2025)", start=dlo, end=dhi),
                    ui.input_action_button("p_go", "Simulate"),
                    ui.hr(),
                    ui.help_text("Uses simple returns derived from log returns. Buy once and hold; equal-weight at start if checked and inverse-price static weight if unchecked."),
                    ui.hr(),
                    ui.help_text("Educational use only — not financial advice."),
                ),
                ui.div(
                    ui.output_text("p_summary"),
                    output_widget("p_plot"),
                    ui.output_table("p_tbl"),
                    # ⬇️ add this block
                    ui.layout_columns(
                        output_widget("p_pie_weights_spent"), # $ allocation & implied weights
                        output_widget("p_pie_shares"),        # initial shares
                        output_widget("p_pie_final"),         # end-of-period $ contribution
                        col_widths=(4, 4, 4),  # 4 columns
                        row_classes="gy-3"     # a little vertical space between rows on wrap
                    ),
                    ui.download_button("p_dl", "Download portfolio series (CSV)"),
                ),
            ),
        ),

        # 3) SECTOR EXPLORER
        ui.nav_panel(
            "Sector Explorer",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_checkbox_group("s_sectors", "Sectors", sector_choices, inline=False),
                    ui.input_checkbox("s_equal", "Equal-weight within sector", value=True),
                    ui.input_date_range("s_dater", "Backtest range (Data available: 9/30/2024-9/30/2025)", start=dlo, end=dhi),
                    ui.input_action_button("s_go", "Build sector indices"),
                    ui.hr(),
                    ui.help_text("'Equal-weight within sector' feature is in beta. Currently, it calculates cumulative returns with equal-weights given to each ticker. Future interations will add an alternative weighting strategy when unchecked."),
                    ui.hr(),
                    ui.help_text("Educational use only — not financial advice."),       
                ),
                ui.div(
                    ui.output_text("s_summary"),
                    output_widget("s_plot"),
                    ui.output_table("s_tbl"),
                    ui.download_button("s_dl", "Download sector indices (CSV)"),
                ),
            ),
        ),

        # 1) EVENT STUDY
        ui.nav_menu(
            "Event Study",
            ui.nav_panel("Intro to SEC Filings",
                ui.output_data_frame("sec_desc_tbl"),
            ),
            ui.nav_panel("Sector Study",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.input_selectize("etype", "Event type(s)", event_types, multiple=True, selected=event_types[:1] if event_types else []),
                        ui.input_slider("k", "Event window (trading days)", min=1, max=20, value=5),
                        ui.input_date_range("dater", "Event date range (Data available: 9/30/2024-9/30/2025)", start=dlo, end=dhi),
                        ui.input_checkbox_group("sector", "Sectors", sector_choices, inline=False),
                        ui.input_checkbox("no_overlap", "Exclude overlapping days (co-occurring events)", value=False),
                        ui.input_action_button("go", "Run"),
                        ui.hr(),
                        ui.help_text("Uses average cumulative log returns for each relative day. buy and hold; equal-weight at start if checked and inverse-price static weight if unchecked."),
                        ui.hr(),
                        ui.help_text("Educational use only — not financial advice."),       
                    ),
                    ui.div(
                        ui.output_text("summary"),
                        output_widget("car_plot"),
                        ui.output_table("tbl"),
                        ui.download_button("dl", "Download table (CSV)"),
                    ),
                ),
            ),
            ui.nav_panel("Individual Stock Study",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.input_selectize("ind_etype", "Event type(s)", event_types, multiple=True, selected=event_types[:1] if event_types else []),
                        ui.input_slider("ind_k", "Event window (trading days)", min=1, max=20, value=5),
                        ui.input_date_range("ind_dater", "Event date range (Data available: 9/30/2024-9/30/2025)", start=dlo, end=dhi),
                        ui.input_selectize("ind_tickers", "Pick tickers (1–10)", ticker_choices, multiple=True),
                        ui.input_action_button("ind_go", "Run"),
                        ui.hr(),
                        ui.help_text("Uses average cumulative log returns for each relative day. buy and hold; equal-weight at start if checked and inverse-price static weight if unchecked."),
                        ui.hr(),
                        ui.help_text("Educational use only — not financial advice."),      
                    ),
                    ui.div(
                        ui.output_text("ind_summary"),
                        output_widget("ind_car_plot"),
                        ui.output_table("ind_tbl"),
                        ui.download_button("ind_dl", "Download table (CSV)"),                                           
                    ),
                ),
            ),
        ),           

        ui.nav_menu(
            "Top 100 ETFs",
            ui.nav_panel("Explore top ETFs",
                ui.output_data_frame("etfs_tbl"),                      
            ),
            ui.nav_panel("ETF Table Descriptions",                      
                ui.div(     
                    ui.markdown(etfs_markdown),
                    # Add some padding for better readability
                    {"style": "padding: 2rem;"}
                ),
            ),            
        ), 

        ui.nav_panel(
            "Appendix",
            ui.div(
                ui.markdown(appendix_markdown),
                class_="appendix"
            ),
        ),
   
    )   
)



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Server logic                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def server(input, output, session):
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2) PORTFOLIO SIMULATOR
    # ─────────────────────────────────────────────────────────────────────────
    @reactive.event(input.p_go)
    def _p_params():
        tickers = (input.p_tickers() or [])[:10]  # light cap
        amt = float(input.p_cash() or 10_000)
        eq = bool(input.p_equal())
        dr = input.p_dater()
        d0 = pd.to_datetime(dr[0]).normalize() if dr else pd.to_datetime(dlo)
        d1 = pd.to_datetime(dr[1]).normalize() if dr else pd.to_datetime(dhi)
        return tickers, amt, eq, d0, d1

    @reactive.calc
    @reactive.event(input.p_go)
    def p_panel():
        tickers, amt, eq, d0, d1 = _p_params()
        if not tickers:
            return pd.DataFrame()

        # Slice AE to chosen tickers + dates
        df = AE.loc[
            (AE["ticker"].isin(tickers)) &
            (AE["date"].between(d0, d1)),
            ["date", "ticker", "logret", "close"]
        ].copy()
        if df.empty:
            return pd.DataFrame()

        # Pivot to wide: simple returns per ticker per day
        df["r"] = _simple_returns(df["logret"])
        r_wide = (
            df.pivot_table(index="date", columns="ticker", values="r", aggfunc="mean")
            .sort_index()
            .fillna(0.0)
        )

        # If no usable return data, bail out
        if r_wide.shape[1] == 0:
            return pd.DataFrame()

        # Per-ticker cumulative index (start at 1)
        cum = (1.0 + r_wide).cumprod()
        cum = cum.div(cum.iloc[0].replace(0, np.nan), axis=1).fillna(1.0)

        if eq:
            # --- BUY-AND-HOLD (equal dollars at start, no rebalancing) ---
            n = r_wide.shape[1]
            w0 = pd.Series(1.0 / max(1, n), index=r_wide.columns)  # sum = 1

            # Portfolio wealth path: initial weights applied once to cumulative paths
            wealth_index = (cum.mul(w0, axis=1)).sum(axis=1)  # starts at 1
        else:
            # --- INVERSE-PRICE STATIC WEIGHTS (buy once, fixed by 1/price₀) ---
            first_prices = (
                df.sort_values("date")
                .dropna(subset=["close"])
                .groupby("ticker")["close"].first()
                .reindex(r_wide.columns)
            )

            invp = 1.0 / first_prices.replace(0, np.nan)
            invp = invp / invp.sum()

            # Fallback if all weights are NaN (e.g., missing first prices)
            if not np.isfinite(invp).any():
                invp = pd.Series(1.0 / r_wide.shape[1], index=r_wide.columns)

            invp = invp.fillna(0.0)
            wealth_index = (cum.mul(invp, axis=1)).sum(axis=1)

        # Scale index to dollars and compute daily portfolio return
        wealth = wealth_index * amt
        r_daily = wealth.pct_change().fillna(0.0)
        r_total = wealth / wealth.iloc[0] - 1

        # Also expose per-ticker cumulative indices (start at 1)
        tickers_cum = cum.copy()

        out = (
            pd.concat(
                [wealth.rename("portfolio_$"), 
                 r_daily.rename("portfolio_r_daily"),
                 r_total.rename("portfolio_r_total")],
                axis=1
            )
            .join(tickers_cum.add_prefix("Cumulative return: "), how="left")
        )
        out.index = pd.to_datetime(out.index)
        return out
    
    # ---- Portfolio meta (weights, $spent, shares, final values) -----------------
    @reactive.calc
    @reactive.event(input.p_go)
    def _p_meta():
        """
        Returns dict of Series (index = ticker):
        w0           : initial weights (sum≈1 across usable tickers)
        p0           : first close in window
        spent        : initial dollars allocated = amt * w0
        shares0      : fractional shares = spent / p0
        final_values : end-of-period dollars per ticker = spent * cum_last
        """
        tickers, amt, eq, d0, d1 = _p_params()
        if not tickers:
            return None

        df = AE.loc[
            (AE["ticker"].isin(tickers)) & (AE["date"].between(d0, d1)),
            ["date", "ticker", "logret", "close"],
        ].copy()
        if df.empty:
            return None

        # daily simple returns → wide
        df["r"] = _simple_returns(df["logret"])
        r_wide = (
            df.pivot_table(index="date", columns="ticker", values="r", aggfunc="mean")
            .sort_index()
            .fillna(0.0)
        )
        if r_wide.shape[1] == 0:
            return None

        # cumulative index rebased to 1
        cum = (1.0 + r_wide).cumprod()
        cum = cum.div(cum.iloc[0].replace(0, np.nan), axis=1).fillna(1.0)

        # first close per ticker (for shares & inverse-price)
        p0 = (
            df.sort_values("date")
            .dropna(subset=["close"])
            .groupby("ticker")["close"]
            .first()
            .reindex(r_wide.columns)
        )

        # weights
        if eq:
            n = r_wide.shape[1]
            w0 = pd.Series(1.0 / max(1, n), index=r_wide.columns)
        else:
            invp = 1.0 / p0.replace(0, np.nan)
            if np.isfinite(invp).sum() == 0:
                w0 = pd.Series(1.0 / r_wide.shape[1], index=r_wide.columns)
            else:
                w0 = (invp / invp.sum()).fillna(0.0)

        spent        = (amt * w0).fillna(0.0)
        shares0      = (spent / p0.replace(0, np.nan)).fillna(0.0)
        cum_last     = cum.iloc[-1]
        final_values = (spent * cum_last).fillna(0.0)

        # keep original column order for stable labeling
        idx = r_wide.columns
        return {
            "w0": w0.loc[idx],
            "p0": p0.loc[idx],
            "spent": spent.loc[idx],
            "shares0": shares0.loc[idx],
            "final_values": final_values.loc[idx],
        }

    # Reactive color map for the current portfolio selection
    @reactive.calc
    def _p_color_map():
        """Compute a stable ticker→color mapping for the current selection."""
        m = _p_meta()
        if not m:
            return {}
        # ⚠️ If your _p_meta uses 'shares0'/'final_values', mirror those names here:
        labels = (pd.Index(m["spent"].index)
                .union(m["shares0"].index)
                .union(m["final_values"].index))
        return _build_color_map(labels)


    @output
    @render_widget
    @reactive.event(input.p_go)
    def p_pie_weights_spent():
        m = _p_meta()
        if not m:
            return None
        cmap = _p_color_map()
        s = m["spent"]
        return _pie_fig(
            names=s.index, values=s.values,
            title="Initial $ allocation (weights)",
            unit="currency", percent=True,
            color_map=cmap,
        )

    @output
    @render_widget
    @reactive.event(input.p_go)
    def p_pie_shares():
        m = _p_meta()
        if not m:
            return None
        cmap = _p_color_map()
        s = m["shares0"] # shares bought at start
        return _pie_fig(
            names=s.index, values=s.values,
            title="Initial shares",
            unit="shares", percent=True, # percent of shares isn’t very meaningful
            color_map=cmap,
        )


    @output
    @render_widget
    @reactive.event(input.p_go)
    def p_pie_final():
        m = _p_meta()
        if not m:
            return None
        cmap = _p_color_map()
        s = m["final_values"] # end-of-period $ per ticker
        return _pie_fig(
            names=s.index, values=s.values,
            title="Final value by ticker",
            unit="currency", percent=True,
            color_map=cmap,
        )
    

    @output
    @render.text
    @reactive.event(input.p_go)
    def p_summary():
        df = p_panel()
        if df.empty:
            return "No data for the selected tickers / date range."
        start_val = float(df["portfolio_$"].iloc[0])
        end_val   = float(df["portfolio_$"].iloc[-1])
        total_ret = (end_val / start_val - 1.0) if start_val else 0.0
        return (
            f"Portfolio start → end: ${start_val:,.0f} → ${end_val:,.0f}  |  "
            f"Total return: {total_ret*100:,.1f}%  |  Days: {len(df)}"
        )

    @output
    @render_widget
    @reactive.event(input.p_go)
    def p_plot():
        df = p_panel()
        fig = go.Figure()

        if df.empty:
            fig.add_annotation(
                text="No data",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14)
            )
            fig.update_layout(template="plotly_white")
            return fig

        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["portfolio_$"],
                mode="lines",
                name="Portfolio",
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Assets Under Management ($)",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            title_x = 0.5,
            template="plotly_white",
            hovermode="x unified",
            margin=dict(t=60, r=20, b=40, l=60),
        )
        fig.update_yaxes(tickprefix="$", separatethousands=True)
        return fig

    @output
    @render.table
    @reactive.event(input.p_go)
    def p_tbl():
        df = p_panel()
        if df.empty:
            return pd.DataFrame()
        # Show last values for the portfolio + per-ticker cumulative indices
        last = df.tail(1).drop(columns=["portfolio_r_daily", "portfolio_r_total", "portfolio_$"]).T.reset_index()
        last.columns = ["Individual stock contributions", "Cumulative return index (start = 1)"]
        return last

    @output
    @render.download(filename=lambda: "portfolio_series.csv")
    @reactive.event(input.p_go)
    def p_dl():
        df = p_panel()
        yield (df.reset_index().rename(columns={"index": "date"}).to_csv(index=False).encode())

    # ─────────────────────────────────────────────────────────────────────────
    # 3) SECTOR EXPLORER
    # ─────────────────────────────────────────────────────────────────────────
    @reactive.event(input.s_go)
    def _s_params():
        secs = input.s_sectors() or []
        eq = bool(input.s_equal())
        dr = input.s_dater()
        d0 = pd.to_datetime(dr[0]).normalize() if dr else pd.to_datetime(dlo)
        d1 = pd.to_datetime(dr[1]).normalize() if dr else pd.to_datetime(dhi)
        return secs, eq, d0, d1

    @reactive.calc
    @reactive.event(input.s_go)
    def s_panel():
        secs, eq, d0, d1 = _s_params()
        if not secs:
            return pd.DataFrame()

        # Slice to selected sectors + dates
        need_cols = ["date", "ticker", "sector", "logret"]
        if not set(need_cols).issubset(AE.columns):
            return pd.DataFrame()
        df = AE.loc[
            (AE["sector"].isin(secs)) &
            (AE["date"].between(d0, d1)),
            need_cols
        ].copy()
        if df.empty:
            return pd.DataFrame()

        # df["r"] = _simple_returns(df["logret"]) (DEPRICATED DUE TO ISSUES WITH HEALTHCARE SECTOR SPLIT-RELATED INDEX EXPLOSION)
        df["r"] = _simple_returns(df["logret"]).astype("float64")
        df["r"].replace([np.inf, -np.inf], np.nan, inplace=True)

        # 1) Neutralize split days if you have flags
        if {"is_split_day", "is_reverse_split_day"}.issubset(AE.columns):
            # Map (date,ticker) → split flags, align, and null those returns
            flags = (
                AE.loc[:, ["date", "ticker", "is_split_day", "is_reverse_split_day"]]
                .set_index(["date", "ticker"])
            )
            m = df.set_index(["date", "ticker"]).index
            split_hits = flags.reindex(m).fillna(False).any(axis=1).to_numpy()
            df.loc[split_hits, "r"] = np.nan

        # 2) De-dup (date,ticker) before sector averaging
        df = (
            df.groupby(["date", "ticker", "sector"], as_index=False, observed=True)["r"]
            .mean()
        )

        # 3) (Optional but practical) Winsorize extremes that slip through
        df["r"] = df["r"].clip(lower=-0.95, upper=0.95)


        # # ---- DIAGNOSTICS (after r computed) ----
        # if "Healthcare" in df["sector"].dropna().unique():
        #     dfh = df.loc[df["sector"] == "Healthcare", ["date","ticker","r"]].copy()
        #     bad_nonfinite = dfh.loc[~np.isfinite(dfh["r"])]
        #     bad_big_pos  = dfh.loc[dfh["r"] > 1]
        #     bad_big_neg  = dfh.loc[dfh["r"] < -0.95]

        #     print("\n[DBG] Healthcare r describe:\n", dfh["r"].describe(), flush=True)
        #     print("[DBG] Non-finite r rows:", len(bad_nonfinite), flush=True)
        #     print("[DBG] r > 1 (top 10):\n", bad_big_pos.sort_values("r", ascending=False).head(10), flush=True)
        #     print("[DBG] r < -0.95 (bottom 10):\n", bad_big_neg.sort_values("r", ascending=True).head(10), flush=True)


        # Option A: equal-weight within sector (average across tickers each day)
        if eq:
            sec_daily = (
                df.groupby(["date", "sector"], observed=True)["r"]
                  .mean()
                  .unstack("sector")
                  .sort_index()
                  .fillna(0.0)
            )
        else:
            # Price or cap weights aren’t guaranteed here; fallback to equal-weight
            sec_daily = (
                df.groupby(["date", "sector"], observed=True)["r"]
                  .mean()
                  .unstack("sector")
                  .sort_index()
                  .fillna(0.0)
            )

        # # The “sec_daily Healthcare top/bottom” block
        # if "Healthcare" in sec_daily.columns:
        #     hc = sec_daily["Healthcare"]
        #     print("[DBG] sec_daily Healthcare top 10 days:\n",
        #         hc.sort_values(ascending=False).head(10), flush=True)
        #     print("[DBG] sec_daily Healthcare bottom 10 days:\n",
        #         hc.sort_values(ascending=True).head(10), flush=True)

        # Build cumulative index for each sector (start at 1)
        sec_cum = (1.0 + sec_daily.fillna(0.0)).cumprod()

        # …and REBASE so the first row is exactly 1.0 for every sector.
        sec_cum = sec_cum.div(sec_cum.iloc[0].replace(0, np.nan), axis=1).fillna(1.0)

        # # ---- TEMP DEBUG (force-flush so you can see it) ----
        # if not sec_cum.empty:
        #     print("\n[s_panel] sec_daily head:\n", sec_daily.head(3), flush=True)
        #     print("[s_panel] sec_cum head:\n", sec_cum.head(3), flush=True)
        #     print("[s_panel] first row of sec_cum:\n", sec_cum.iloc[0], flush=True)
        # # ---- END DEBUG ----
        # print(f"[s_panel] secs={secs}, eq={eq}, d0={d0.date()} d1={d1.date()}", flush=True)

        return sec_cum
    

    @output
    @render.text
    @reactive.event(input.s_go)
    def s_summary():
        df = s_panel()
        if df.empty:
            return "No data for the selected sectors / date range."
        cols = list(df.columns)
        return f"Sectors: {', '.join(cols)}  |  Days: {len(df)}"

    @output
    @render_widget
    @reactive.event(input.s_go)
    def s_plot():
        df = s_panel()
        fig = go.Figure()

        if df.empty:
            fig.add_annotation(
                text="No data",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14)
            )
            fig.update_layout(template="plotly_white", title_x=0.5)
            return fig
        
        # One line per sector
        for c in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df[c],
                    mode="lines", name=c,
                    hovertemplate="%{x|%Y-%m-%d}<br>Index: %{y:.3f}x<extra></extra>",
                )
            )

        fig.update_layout(
            title="Sector Cumulative Indices (start = 1)",
            title_x=0.5,
            template="plotly_white",
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Index (start = 1x)",
            margin=dict(t=60, r=20, b=40, l=60),
            legend_title_text="Sector",
        )
        return fig

    @output
    @render.table
    @reactive.event(input.s_go)
    def s_tbl():
        df = s_panel()
        if df.empty:
            return pd.DataFrame()
        last = df.tail(1).T.reset_index()
        last.columns = ["Sector", "Cumulative return index (start = 1)"]
        last["Total return"] = last["Cumulative return index (start = 1)"] - 1.0
        # nice rounding for display
        last["Cumulative return index (start = 1)"] = last["Cumulative return index (start = 1)"].round(3)
        last["Total return"] = (last["Total return"] * 100).round(2).astype(str) + "%"
        return last

    @output
    @render.download(filename=lambda: "sector_indices.csv")
    @reactive.event(input.s_go)
    def s_dl():
        df = s_panel()
        yield (df.reset_index().rename(columns={"index": "date"}).to_csv(index=False).encode())


    # ─────────────────────────────────────────────────────────────────────────
    # 1) EVENT STUDY
    # ─────────────────────────────────────────────────────────────────────────

    @reactive.event(input.go)
    def _params():
        """
        Returns:
            sel_types (list[str])  : event types selected by user
            k        (int)         : window size
            date_lo/hi (Timestamp) : event date range
            sectors  (list[str])   : sector filters (may be empty)
            no_overlap (bool)      : drop days with multiple event types per ticker
        """
        sel_types = input.etype() or []
        k = int(input.k())
        dr = input.dater()
        date_lo = pd.to_datetime(dr[0]).normalize() if dr else pd.Timestamp("2022-01-01")
        date_hi = pd.to_datetime(dr[1]).normalize() if dr else pd.Timestamp(date.today())
        sectors = input.sector() or []
        no_overlap = bool(input.no_overlap())
        return sel_types, k, date_lo, date_hi, sectors, no_overlap

    @reactive.calc
    @reactive.event(input.go)
    def panel() -> pd.DataFrame:
        """
        Build ±k trading-day windows around each selected event *type*.
        Returns tidy columns:
            ['ticker','event_type','event_date','date','rel_day','logret']
        """
        sel_types, k, date_lo, date_hi, sectors, no_overlap = _params()

        # 1) Filter EVENTS by user selections
        ev = EVENTS.copy()
        mask = ev["date"].between(date_lo, date_hi)
        if sel_types:
            mask &= ev["event_type"].isin(sel_types)
        if sectors and "sector" in ev.columns:
            mask &= ev["sector"].isin(sectors)
        if no_overlap:
            mask &= ~ev["is_overlap"]

        ev = ev.loc[mask, ["ticker", "date", "event_type"]].rename(columns={"date": "event_date"})
        if ev.empty:
            return pd.DataFrame()
        print(f"ev_masked: {ev}")

        # 2) Map each (ticker, event_date) to its trading-day index (event_tidx)
        idx_map = AE.loc[:, ["ticker", "date", "tidx"]].rename(columns={"date": "event_date", "tidx": "event_tidx"})
        ev = ev.merge(idx_map, on=["ticker", "event_date"], how="left").dropna(subset=["event_tidx"])
        if ev.empty:
            return pd.DataFrame()
        print(f"ev_idx_mapped: {ev}")

        # 3) Cross-join with offsets −k..+k
        offsets = pd.DataFrame({"rel_day": np.arange(-k, k + 1, dtype=np.int16)})
        evx = ev.merge(offsets, how="cross")
        evx["tidx"] = evx["event_tidx"] + evx["rel_day"]
        print(f"evx: {evx}")

        # 4) Join back to AE to fetch actual date & log returns
        base = AE.loc[:, ["ticker", "tidx", "date", "logret"]]
        out = (
            evx.merge(base, on=["ticker", "tidx"], how="left")
                .dropna(subset=["date"])
                .sort_values(["ticker", "event_date", "date"])
                .loc[:, ["ticker", "event_type", "event_date", "date", "rel_day", "logret"]]
        )
        print(f"out: {out}")
        return out

    @output
    @render.text
    @reactive.event(input.go)
    def summary():
        """Human-friendly summary of the current run"""
        df = panel()
        if df.empty:
            return f"No events found. • Source: {DATA_PQ.name}"
        n_ev = df[["ticker", "event_date", "event_type"]].drop_duplicates().shape[0]
        by_rel = df["rel_day"].value_counts().sort_index()
        return (
            f"Events matched: {n_ev} | Rows in ±k: {len(df):,} • "
            f"Min/Max rows per rel_day: {int(by_rel.min())}/{int(by_rel.max())} • "
            f"Source: {DATA_PQ.name}"
        )

    @output
    @render.table
    @reactive.event(input.go)
    def tbl():
        """Aggregate stats by relative day: mean/median/count of log returns"""
        df = panel()
        if df.empty:
            return pd.DataFrame()
        return (
            df.groupby("rel_day")["logret"]
              .agg(mean="mean", median="median", count="count")
              .reset_index()
        )

    @output
    @render_widget
    @reactive.event(input.go)
    def car_plot():
        """
        Plot Average CAR with a ±95% CI ribbon using Plotly.
        CAR = cumulative sum of mean log returns across rel_day.
        """
        df = panel()
        fig = go.Figure()

        if df.empty:
            fig.add_annotation(
                text="No events selected",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14)
            )
            fig.update_layout(template="plotly_white")
            return fig

        g = df.groupby("rel_day")["logret"]
        mu = g.mean().sort_index()                                # mean return per rel_day
        n  = g.count().sort_index()                               # sample size per rel_day
        se = g.std(ddof=1).sort_index() / np.sqrt(n.clip(lower=1))  # standard error

        car    = mu.cumsum()
        car_lo = (mu - 1.96 * se).cumsum()
        car_hi = (mu + 1.96 * se).cumsum()

        xvals = car.index.values

        # after you compute mu, n, se, car, car_lo, car_hi
        xvals = car.index.astype(int)

        fig = go.Figure()


        # CI ribbon: add upper, then lower with fill='tonexty'
        # CI ribbon
        ci_rgba = "rgba(31, 119, 180, 0.20)"   # 20% opacity; tweak last number (0–1)

        # upper bound (no visible line)
        fig.add_trace(go.Scatter(
            x=xvals, y=car_hi.values,
            line=dict(width=0, color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False, name="CI hi"
        ))

        # lower bound + fill to previous
        fig.add_trace(go.Scatter(
            x=xvals, y=car_lo.values,
            fill="tonexty",
            fillcolor=ci_rgba,                  # <<< controls ribbon opacity
            line=dict(width=0, color="rgba(0,0,0,0)"),
            name="95% CI", showlegend=True, hoverinfo="skip"
        ))

        # main CAR line (with % shown in hover via customdata)
        fig.add_trace(go.Scatter(
            x=xvals, y=car.values, mode="lines", name="Avg CAR",
            customdata=np.expm1(car.values),  # ≈ percent return
            meta=n.values,                    # sample size at each rel_day
            hovertemplate=(
                "Day %{x}<br>"
                "CAR (log): %{y:.4f}<br>"
                "CAR (≈%): %{customdata:.2%}<br>"
                "n: %{meta}<extra></extra>"
            )
        ))


        # Event-day vertical line at 0
        fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.6)

        fig.update_layout(
            title="Average Cumulative Abnormal Log Return (±95% CI)",
            xaxis_title="Relative day",
            yaxis_title="Avg CAR (log units)",
            title_x = 0.5,
            template="plotly_white",
            hovermode="x unified",  # track single x position
            margin=dict(t=60, r=20, b=40, l=60),
            legend=dict(itemsizing="constant"),
        )

        return fig

    @output
    @render.download(filename=lambda: "event_windows.csv")
    @reactive.event(input.go)
    def dl():
        """Download the raw (ticker, event_date, date, rel_day, logret) panel as CSV."""
        df = panel()
        if df.empty:
            # Returning an empty payload avoids a broken download
            yield b""
            return
        yield df.to_csv(index=False).encode()


    # ─────────────────────────────────────────────────────────────────────────
    # 1B) EVENT STUDY — INDIVIDUAL STOCKS
    # ─────────────────────────────────────────────────────────────────────────

    @reactive.event(input.ind_go)
    def _ind_params():
        """
        Returns:
            sel_types (list[str])  : event types (forms) selected by user
            k        (int)         : window size for ±k trading days
            date_lo/hi (Timestamp) : event date filter
            tickers  (list[str])   : which tickers to include
        """
        sel_types = input.ind_etype() or []
        k = int(input.ind_k())
        dr = input.ind_dater()
        date_lo = pd.to_datetime(dr[0]).normalize() if dr else pd.Timestamp("2022-01-01")
        date_hi = pd.to_datetime(dr[1]).normalize() if dr else pd.Timestamp(date.today())
        tickers = input.ind_tickers() or []
        return sel_types, k, date_lo, date_hi, tickers

    @reactive.calc
    @reactive.event(input.ind_go)
    def ind_panel() -> pd.DataFrame:
        """
        Build ±k trading-day windows around each selected event *type*,
        restricted to the chosen tickers.

        Returns tidy columns:
            ['ticker','event_type','event_date','date','rel_day','logret']
        """
        sel_types, k, date_lo, date_hi, tickers = _ind_params()

        # 1) Filter EVENTS by user selections
        ev = EVENTS.copy()
        mask = ev["date"].between(date_lo, date_hi)

        if sel_types:
            mask &= ev["event_type"].isin(sel_types)

        if tickers:
            mask &= ev["ticker"].isin(tickers)

        ev = (
            ev.loc[mask, ["ticker", "date", "event_type"]]
              .rename(columns={"date": "event_date"})
        )
        if ev.empty:
            return pd.DataFrame()

        # 2) Map each (ticker, event_date) to its trading-day index (event_tidx)
        idx_map = (
            AE.loc[:, ["ticker", "date", "tidx"]]
              .rename(columns={"date": "event_date", "tidx": "event_tidx"})
        )
        ev = (
            ev.merge(idx_map, on=["ticker", "event_date"], how="left")
              .dropna(subset=["event_tidx"])
        )
        if ev.empty:
            return pd.DataFrame()

        # 3) Cross-join with offsets −k..+k
        offsets = pd.DataFrame({"rel_day": np.arange(-k, k + 1, dtype=np.int16)})
        evx = ev.merge(offsets, how="cross")
        evx["tidx"] = evx["event_tidx"] + evx["rel_day"]

        # 4) Join back to AE to fetch actual date & log returns
        base = AE.loc[:, ["ticker", "tidx", "date", "logret"]]
        out = (
            evx.merge(base, on=["ticker", "tidx"], how="left")
               .dropna(subset=["date"])
               .sort_values(["ticker", "event_date", "date"])
               .loc[:, ["ticker", "event_type", "event_date", "date", "rel_day", "logret"]]
        )
        return out
    
    @output
    @render.text
    @reactive.event(input.ind_go)
    def ind_summary():
        """Human-friendly summary for the Individual Stocks study"""
        df = ind_panel()
        if df.empty:
            return "No events found for the selected tickers / settings."

        n_ev = df[["ticker", "event_date", "event_type"]].drop_duplicates().shape[0]
        by_rel = df["rel_day"].value_counts().sort_index()
        tickers_used = ", ".join(sorted(df["ticker"].unique().tolist()))

        return (
            f"Tickers: {tickers_used} | "
            f"Events matched: {n_ev} | Rows in ±k: {len(df):,} • "
            f"Min/Max rows per rel_day: {int(by_rel.min())}/{int(by_rel.max())}"
        )
    
    @output
    @render.table
    @reactive.event(input.ind_go)
    def ind_tbl():
        """
        Aggregate stats by relative day (across selected tickers):
        mean / median / count of log returns.
        """
        df = ind_panel()
        if df.empty:
            return pd.DataFrame()

        return (
            df.groupby("rel_day")["logret"]
              .agg(mean="mean", median="median", count="count")
              .reset_index()
        )
    
    @output
    @render_widget
    @reactive.event(input.ind_go)
    def ind_car_plot():
        """
        Interactive Average CAR (cumulative mean log return across events
        for the selected tickers) with a ±95% CI ribbon.
        """
        df = ind_panel()
        fig = go.Figure()

        if df.empty:
            fig.add_annotation(
                text="No events selected",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14)
            )
            fig.update_layout(template="plotly_white")
            return fig

        g  = df.groupby("rel_day")["logret"]
        mu = g.mean().sort_index()                                # mean per rel_day
        n  = g.count().sort_index()                               # sample size
        se = g.std(ddof=1).sort_index() / np.sqrt(n.clip(lower=1))  # standard error

        car    = mu.cumsum()                 # cumulative mean
        car_lo = (mu - 1.96 * se).cumsum()   # lower CI
        car_hi = (mu + 1.96 * se).cumsum()   # upper CI
        xvals  = car.index.values


        # after you compute mu, n, se, car, car_lo, car_hi
        xvals = car.index.astype(int)

        fig = go.Figure()

        # CI ribbon: add upper, then lower with fill='tonexty'
        # CI ribbon
        ci_rgba = "rgba(31, 119, 180, 0.20)"   # 20% opacity; tweak last number (0–1)

        # upper bound (no visible line)
        fig.add_trace(go.Scatter(
            x=xvals, y=car_hi.values,
            line=dict(width=0, color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False, name="CI hi"
        ))

        # lower bound + fill to previous
        fig.add_trace(go.Scatter(
            x=xvals, y=car_lo.values,
            fill="tonexty",
            fillcolor=ci_rgba,                  # <<< controls ribbon opacity
            line=dict(width=0, color="rgba(0,0,0,0)"),
            name="95% CI", showlegend=True, hoverinfo="skip"
        ))

        # main CAR line (with % shown in hover via customdata)
        fig.add_trace(go.Scatter(
            x=xvals, y=car.values, mode="lines", name="Avg CAR",
            customdata=np.expm1(car.values),  # ≈ percent return
            meta=n.values,                    # sample size at each rel_day
            hovertemplate=(
                "Day %{x}<br>"
                "CAR (log): %{y:.4f}<br>"
                "CAR (≈%): %{customdata:.2%}<br>"
                "n: %{meta}<extra></extra>"
            )
        ))

        # Event-day vertical line
        fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.6)

        fig.update_layout(
            title="Average Cumulative Abnormal Log Return (±95% CI)",
            xaxis_title="Relative day",
            yaxis_title="Avg CAR (selected tickers)",
            title_x = 0.5,
            template="plotly_white",
            hovermode="x",
            margin=dict(t=60, r=20, b=40, l=60),
            legend=dict(itemsizing="constant"),
        )
        return fig
    
    @output
    @render.download(filename=lambda: "individual_event_windows.csv")
    @reactive.event(input.ind_go)
    def ind_dl():
        """
        Download the raw (ticker, event_type, event_date, date, rel_day, logret)
        panel for the Individual Stocks study as CSV.
        """
        df = ind_panel()
        if df.empty:
            yield b""
            return
        yield df.to_csv(index=False).encode()

    @output
    @render.data_frame
    def sec_desc_tbl():
        df = pd.read_csv(SEC_CSV)
        return render.DataTable(
            df,
            filters=True,       # column filters
            selection_mode="row",
            height="80vh",     # scroll if tall
            width="100%"
        )
    
    @output
    @render.data_frame
    def etfs_tbl():
        df = pd.read_csv(ETF_CSV)
        return render.DataTable(
            df,
            filters=True,       # column filters
            selection_mode="row",
            height="80vh",     # scroll if tall
            width="100%"
        )

# Bundle UI + server into a Shiny app
app = App(app_ui, server, static_assets=www_dir)
