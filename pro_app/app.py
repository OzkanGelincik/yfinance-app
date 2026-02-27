"""
app.py — Event Study Studio (DuckDB + S3)
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
from shinywidgets import output_widget, render_widget   # <-- add this
# (In the UI we’ll use ui.output_widget(...))
import plotly.graph_objects as go
import plotly.express as px
import duckdb
import os 

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Locate & load dataset                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Folder where this app.py lives (robust to "current working directory" issues) 
APP_DIR = Path(__file__).resolve().parent

OUT_DIR = APP_DIR / "outputs"

SEC_CSV = OUT_DIR / "sec_filing_descriptions.csv"

ETF_CSV = OUT_DIR / "top_100_etfs_described.csv"

TICKERS_CSV = OUT_DIR / "tickers.csv"

www_dir = Path(__file__).parent / "www"

SEC_DESC_DF = pd.read_csv(SEC_CSV)

ETF_DF      = pd.read_csv(ETF_CSV)



def _load_ticker_universe(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Create it once (distinct tickers) and place it in outputs/."
        )
    df = pd.read_csv(path)
    # Accept either column name "ticker" or first column
    col = "ticker" if "ticker" in df.columns else df.columns[0]
    tickers = (
        df[col]
        .astype(str)
        .str.upper()
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )
    tickers.sort()
    return tickers

TICKER_UNIVERSE = _load_ticker_universe(TICKERS_CSV)



# Optional: load local env vars for dev (DO NOT commit .env)
# pip install python-dotenv
try:
    from dotenv import load_dotenv
    load_dotenv(APP_DIR / ".env", override=False)
except Exception:
    pass


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DuckDB + S3 hookup                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝


S3_URI = os.getenv(
    "AE_S3_URI",
    "https://yfinance-app-data.s3.us-east-1.amazonaws.com/yfinance-app/analysis_enriched_backfilled_v7_3y_11col_tidx_year.parquet",
)

def _configure_duckdb_s3(con: duckdb.DuckDBPyConnection) -> None:
    """
    Safe S3 credential loading for DuckDB.

    Order:
      1) DuckDB aws credential chain (reads AWS env vars, ~/.aws/*, IAM role/web identity, etc.)
      2) Explicit env var fallback (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN)

    Nothing is hardcoded. Nothing is written to disk.
    """
    # Extensions (INSTALL is optional if already available in your environment)
    def _safe_install_load(con, ext: str):
        try:
            con.execute(f"INSTALL {ext};")
        except Exception:
            pass
        con.execute(f"LOAD {ext};")

    _safe_install_load(con, "httpfs")
    _safe_install_load(con, "aws")

    # Region can come from env; default is fine for your bucket
    aws_region = os.getenv("AWS_REGION", "us-east-1")
    con.execute(f"SET s3_region='{aws_region}';")

    # Prefer DuckDB's AWS credential chain (recommended)
    # This will pick up (in order, broadly): env vars, shared config/credentials files, IAM role, web identity, etc.
    try:
        con.execute("CALL load_aws_credentials();")
        return
    except Exception:
        # Fall through to explicit env var config if you supplied them
        pass

    # Fallback: explicit env vars (still safe, because they are not committed)
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("AWS_SESSION_TOKEN")  # optional (for STS)

    if access_key and secret_key:
        def _q(s: str) -> str:
            return s.replace("'", "''")

        con.execute(f"SET s3_access_key_id='{_q(access_key)}';")
        con.execute(f"SET s3_secret_access_key='{_q(secret_key)}';")
        if session_token:
            con.execute(f"SET s3_session_token='{_q(session_token)}';")
        return

    # If we got here, there are no usable credentials
    raise RuntimeError(
        "No AWS credentials found. Provide credentials via one of these safe methods:\n"
        "  - AWS default credential chain (recommended): AWS_PROFILE / ~/.aws/credentials / IAM role\n"
        "  - Or env vars: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (optional AWS_SESSION_TOKEN)\n"
    )


def _duckdb_connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    _configure_duckdb_s3(con)
    return con

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Helpers used by server/plots                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _simple_returns(logret: pd.Series) -> pd.Series:
    return np.expm1(logret.fillna(0.0))

# Plotly helpers: KEEP THESE
def _build_color_map(labels):
    palette = (px.colors.qualitative.D3
               + px.colors.qualitative.Set2
               + px.colors.qualitative.Plotly)
    labs = sorted(pd.Index(labels).unique())
    return {lab: palette[i % len(palette)] for i, lab in enumerate(labs)}

def _pie_fig(*, names, values, title, hole=0.35, unit=None, percent=True, color_map=None):
    ser = pd.Series(values, index=pd.Index(names, name="ticker"))
    if color_map:
        ordered = [t for t in color_map.keys() if t in ser.index]
        ser = ser.reindex(ordered)

    fig = px.pie(
        names=ser.index,
        values=ser.values,
        hole=hole,
        title=title,
        color=ser.index,
        color_discrete_map=color_map or {},
    )

    if unit == "currency":
        ht = "%{label}: $%{value:,.0f}"
    elif unit == "shares":
        ht = "%{label}: %{value:,.4f} sh"
    else:
        ht = "%{label}: %{value}"
    if percent:
        ht += " (%{percent})"

    textinfo = "percent" if percent else "none"
    fig.update_traces(hovertemplate=ht + "<extra></extra>", textinfo=textinfo)

    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), title_x=0.5, legend_title_text="Ticker")
    return fig

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  UI choices (computed once, tiny queries)                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

con = _duckdb_connect()

try:
    def _ddb_quote_string(s: str) -> str:
        # DuckDB uses single quotes for string literals
        return "'" + s.replace("\\", "\\\\").replace("'", "''") + "'"

    con.execute(f"CREATE OR REPLACE VIEW ae AS SELECT * FROM read_parquet({_ddb_quote_string(S3_URI)});")

    con.execute("""
    CREATE OR REPLACE VIEW events AS
    WITH filing AS (
      SELECT DISTINCT UPPER(ticker) AS ticker, date::DATE AS date, filing_form AS event_type, sector
      FROM ae
      WHERE filing_form IS NOT NULL
    ),
    splits AS (
      SELECT DISTINCT UPPER(ticker) AS ticker, date::DATE AS date, 'SPLIT' AS event_type, sector
      FROM ae
      WHERE is_split_day = TRUE
    ),
    rsplits AS (
      SELECT DISTINCT UPPER(ticker) AS ticker, date::DATE AS date, 'REVERSE_SPLIT' AS event_type, sector
      FROM ae
      WHERE is_reverse_split_day = TRUE
    ),
    all_events AS (
      SELECT * FROM filing
      UNION ALL SELECT * FROM splits
      UNION ALL SELECT * FROM rsplits
    ),
    counts AS (
      SELECT ticker, date, COUNT(*) AS n_events
      FROM all_events
      GROUP BY 1,2
    )
    SELECT e.*, c.n_events, (c.n_events > 1) AS is_overlap
    FROM all_events e
    JOIN counts c USING (ticker, date);
    """)

    def ddb_df(sql: str, params=None) -> pd.DataFrame:
        return con.execute(sql, params or []).df()

    sector_choices = ddb_df("""
        SELECT DISTINCT TRIM(sector) AS sector
        FROM ae
        WHERE sector IS NOT NULL AND TRIM(sector) <> ''
        ORDER BY sector;
    """)["sector"].tolist()

    event_types = ddb_df("""
        SELECT DISTINCT event_type
        FROM events
        ORDER BY event_type;
    """)["event_type"].tolist()

    bounds = ddb_df("SELECT MIN(date::DATE) AS min_date, MAX(date::DATE) AS max_date FROM ae;").iloc[0]
    hi = pd.to_datetime(bounds["max_date"]).normalize()
    lo = max(pd.to_datetime(bounds["min_date"]).normalize(), hi - pd.Timedelta(days=756))
    dlo, dhi = str(lo.date()), str(hi.date())


    # After: dlo, dhi = str(lo.date()), str(hi.date())

    def _fmt_mdy(iso_yyyy_mm_dd: str) -> str:
        d = pd.to_datetime(iso_yyyy_mm_dd).date()
        return f"{d.month}/{d.day}/{d.year}"

    DATA_AVAIL_TEXT = f"Data available: {_fmt_mdy(dlo)} to {_fmt_mdy(dhi)}"

finally:
    con.close()



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
- **Caching/IO** - Parquet for fast reloads; local JSON cache for SEC responses; CSV export from the app.
- **DuckDB (query engine) + S3 Parquet (storage), pandas for transforms, Plotly for charts.**

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


#### **Step 10: Build the “Pro” thin dataset for the Shiny app (from v7)**
The legacy app loaded a large Parquet dataset into pandas at startup, which limited how much history we could ship and often caused deployment memory failures.  
For the **pro** app, we generate a **thin, app-optimized Parquet** derived from the full v7 meta dataset and host it on **Amazon S3**. The app then uses **DuckDB** to query only the rows and columns needed for each user action (portfolio, sectors, event windows), instead of loading everything into memory.

- **Source (meta):** `analysis_enriched_backfilled_v7.parquet` (wide schema, full enrichment)
- **Output (pro thin):** `analysis_enriched_backfilled_v7_3y_11col_tidx_year.parquet` (thin schema, app-ready)
- **Storage:** Amazon S3 (Parquet)
- **Runtime access:** DuckDB `read_parquet(...)` directly against S3 (no full dataset download into app memory)
- **Config:** S3 object is controlled by `AE_S3_URI` (env var). A default URI is provided for convenience.

#### **Step 10A: Filter the row scope to a multi-year window**
We restrict the dataset to a **recent multi-year window** (about the last ~3 years) to keep the pro app responsive while still supporting:
- longer backtests (useful for checking seasonality patterns)
- more event observations (more robust event study statistics)

- **Window:** ~3 years of daily rows per ticker (where price history exists)
- **Why:** more history than the legacy app, without shipping a heavy dataset into memory

#### **Step 10B: Keep only the columns required by the pro app**
The pro dataset keeps a minimal schema that supports all three Shiny workflows while staying lightweight for cloud reads.

**Columns kept (pro thin schema):**
- `date` (trading date)
- `ticker` (uppercased symbol)
- `close` (price used in app displays and weighting logic)
- `logret` (log daily return. Converted to simple returns in Portfolio/Sector. Used directly in Event Study)
- `filing_form` (SEC filing event label for event selection)
- `sector` (sector filters and sector indexing)
- `is_split_day` (forward split event flag)
- `is_reverse_split_day` (reverse split event flag)
- `market_cap` (optional summaries/filters where relevant)
- `tidx` (per-ticker trading-day index for fast ±k event windows)
- `year` (precomputed helper for fast filtering/grouping and to avoid repeated date parsing)

#### **Step 10C: Add trading-day indexing for fast and correct event windows**
To construct true **±k trading-day** windows (ignoring weekends and market holidays), we add:

- `tidx = 0, 1, 2, ...` within each ticker’s trading calendar

Event windows are built by joining on `tidx` offsets (e.g., `event_tidx + rel_day`), which avoids expensive calendar logic and keeps event window construction fast in DuckDB.

#### **Step 10D: Host the thin dataset on S3 and query it at runtime**
The final thin dataset is stored in S3 and queried directly by DuckDB during app usage.

**Final File (pro thin dataset):**
- `analysis_enriched_backfilled_v7_3y_11col_tidx_year.parquet`

---

## **Pro thin dataset dictionary (what the deployed app actually queries)**

| Variable | dtype | Description |
| :--- | :--- | :--- |
| `date` | date / datetime | Trading date (YYYY-MM-DD). |
| `ticker` | string | Stock symbol (uppercased). |
| `close` | float | Close price used for portfolio weighting and displays. |
| `logret` | float | Log daily return. Used directly in Event Study; converted to simple returns in other panels. |
| `market_cap` | float | Approximate market cap (may be missing for some tickers). |
| `sector` | string | Sector classification used for filtering and sector indexing. |
| `filing_form` | string | SEC filing form on that date (event label). Null on non-filing days. |
| `is_split_day` | bool | True on forward split dates. |
| `is_reverse_split_day` | bool | True on reverse split dates. |
| `tidx` | int | Per-ticker trading-day index used to build ±k trading-day windows. |
| `year` | int | Calendar year derived from `date` for fast filtering/grouping. |

---

## **v7 meta dataset dictionary (reference, not loaded by the pro app)**
The full v7 dataset contains additional fields used during enrichment and validation (e.g., OHLCV, identifiers, share history, SIC metadata). The pro app does not load this wide table at runtime.

| Variable | dtype | Description |
| :--- | :--- | :--- |
| `open` | float | Opening price for the trading day. |
| `high` | float | Highest price traded during the day. |
| `low` | float | Lowest price traded during the day. |
| `adj_close` | float | Adjusted close price (accounts for splits and sometimes dividends). |
| `volume` | float / int | Number of shares traded on that day. |
| `ret` | float | Simple daily return. |
| `industry` | string | More granular industry classification. |
| `cik` | string / int-like | SEC CIK identifier. |
| `sic` | string / int-like | SIC code. |
| `sic_desc` | string | SIC description. |
| `shares_outstanding` | float | Shares outstanding (time-varying where available). |
| `float_shares` | float | Float shares (time-varying where available). |
| `free_float` | float | Float ratio. |
| `split_ratio` | float | Split ratio for split events. |
| `split_cum_factor` | float | Cumulative split adjustment factor. |
| `last_filing_date` | date | Last filing date on or before the row’s date. |
| `is_filing_day` | bool | True on filing dates. |
| `days_since_filing` | int | Days since last filing. |
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
<br><br>
- **Bonds (Fixed Income):**  
`Credit Quality/Type: ~% Breakdown by Issuer Type`
    - Example:  
    *"US Inv Grade Bonds: ~43% Treasuries..."* means the fund holds safe US debt, split between Government bonds (Treacheries) and other types.
<br><br>
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
<br><br>
- **Bond Types**
    - **Treasuries:** Debt issued by the US Government (safest).  
    - **MBS:** Mortgage-Backed Securities (pools of home loans, usually agency-backed).  
    - **Corporates:** Debt issued by companies.  
    - **Inv Grade:** Investment Grade (high credit quality, lower risk).  
    - **High Yield / Junk:** Non-investment grade (lower credit quality, higher risk/yield).  
    - **Muni:** Municipal Bonds (issued by states/cities, often tax-exempt).  
<br><br>
- **Regions**
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


app_ui = ui.page_fluid(
    head_links,
    ui.h2("Portfolio, Sector & Event Lab Pro", class_="fw-bold"),
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
                style= "padding: 2rem;",
            ),
        ),


        # 2) PORTFOLIO SIMULATOR
        ui.nav_panel(
            "Portfolio Simulator",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_text("p_ticker_search", "Search tickers", placeholder="Type e.g. aa, nv, msft"), # this line replaced `ui.input_selectize("p_tickers", ..., ticker_choices, multiple=True)`
                    ui.input_selectize("p_matches", "Matches", choices=[], multiple=False), # this line replaced `ui.input_selectize("p_tickers", ..., ticker_choices, multiple=True)`
                    ui.input_action_button("p_add", "Add ticker"), # this line replaced `ui.input_selectize("p_tickers", ..., ticker_choices, multiple=True)`
                    ui.input_action_button("p_clear", "Clear tickers"), # this line replaced `ui.input_selectize("p_tickers", ..., ticker_choices, multiple=True)`
                    ui.output_text("p_selected"), # this line replaced `ui.input_selectize("p_tickers", ..., ticker_choices, multiple=True)`
                    ui.input_numeric("p_cash", "Initial cash ($)", value=10_000, min=100),
                    ui.input_checkbox("p_equal", "Equal-weight portfolio?", value=True),
                    ui.input_date_range("p_dater", f"Backtest range ({DATA_AVAIL_TEXT})", start=dlo, end=dhi, min = dlo, max = dhi),
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
                    ui.input_date_range("s_dater", f"Backtest range ({DATA_AVAIL_TEXT})", start=dlo, end=dhi, min = dlo, max = dhi),
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
                        ui.input_date_range("dater", f"Backtest range ({DATA_AVAIL_TEXT})", start=dlo, end=dhi, min = dlo, max = dhi),
                        ui.input_checkbox_group("sector", "Sectors", sector_choices, inline=False),
                        ui.input_checkbox("no_overlap", "Exclude overlapping days (co-occurring events)", value=False),
                        ui.input_action_button("go", "Run"),
                        ui.hr(),
                        ui.help_text("Uses log returns. Abnormal return assumes expected return = 0."),
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
                        ui.input_date_range("ind_dater", f"Backtest range ({DATA_AVAIL_TEXT})", start=dlo, end=dhi, min = dlo, max = dhi),
                        
                        ui.input_text("ind_ticker_search", "Search tickers", placeholder="Type e.g. aa, nv, msft"), # replaces ui.input_selectize("ind_tickers", "Pick tickers (1–10)", ticker_choices, multiple=True),
                        ui.input_selectize("ind_matches", "Matches", choices=[], multiple=False), # replaces # ui.input_selectize("ind_tickers", "Pick tickers (1–10)", ticker_choices, multiple=True),
                        ui.input_action_button("ind_add", "Add ticker"), # replaces # ui.input_selectize("ind_tickers", "Pick tickers (1–10)", ticker_choices, multiple=True),
                        ui.input_action_button("ind_clear", "Clear tickers"), # replaces # ui.input_selectize("ind_tickers", "Pick tickers (1–10)", ticker_choices, multiple=True),
                        ui.output_text("ind_selected"), # replaces # ui.input_selectize("ind_tickers", "Pick tickers (1–10)", ticker_choices, multiple=True),

                        ui.input_action_button("ind_go", "Run"),
                        ui.hr(),
                        ui.help_text("Uses log returns. Abnormal return assumes expected return = 0."),
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
                    style= "padding: 2rem;",
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
    con_holder = {"con": None}

    def get_session_con() -> duckdb.DuckDBPyConnection:
        if con_holder["con"] is not None:
            return con_holder["con"]

        con = duckdb.connect(database=":memory:")
        _configure_duckdb_s3(con)

        def _ddb_quote_string(s: str) -> str:
            return "'" + s.replace("\\", "\\\\").replace("'", "''") + "'"

        con.execute(
            f"CREATE OR REPLACE VIEW ae AS "
            f"SELECT * FROM read_parquet({_ddb_quote_string(S3_URI)});"
        )

        con.execute("""
        CREATE OR REPLACE VIEW events AS
        WITH filing AS (
          SELECT DISTINCT upper(ticker) AS ticker, date::DATE AS date, filing_form AS event_type, sector
          FROM ae
          WHERE filing_form IS NOT NULL
        ),
        splits AS (
          SELECT DISTINCT upper(ticker) AS ticker, date::DATE AS date, 'SPLIT' AS event_type, sector
          FROM ae
          WHERE is_split_day = TRUE
        ),
        rsplits AS (
          SELECT DISTINCT upper(ticker) AS ticker, date::DATE AS date, 'REVERSE_SPLIT' AS event_type, sector
          FROM ae
          WHERE is_reverse_split_day = TRUE
        ),
        all_events AS (
          SELECT * FROM filing
          UNION ALL SELECT * FROM splits
          UNION ALL SELECT * FROM rsplits
        ),
        counts AS (
          SELECT ticker, date, COUNT(*) AS n_events
          FROM all_events
          GROUP BY 1,2
        )
        SELECT e.*, c.n_events, (c.n_events > 1) AS is_overlap
        FROM all_events e
        JOIN counts c USING (ticker, date);
        """)

        con_holder["con"] = con

        def _cleanup():
            try:
                con.close()
            except Exception:
                pass
            con_holder["con"] = None

        session.on_ended(_cleanup)
        return con
    
    # ─────────────────────────────────────────────────────────────────────────
    # 1) PORTFOLIO SIMULATOR (DuckDB-backed)
    # ─────────────────────────────────────────────────────────────────────────

    def _search_tickers(prefix: str, limit: int = 25) -> list[str]:
        prefix = (prefix or "").strip().upper()
        # Minimum length helps avoid returning huge match lists and reduces churn
        if len(prefix) < 2:
            return []

        # Fast in-memory prefix match
        out = [t for t in TICKER_UNIVERSE if t.startswith(prefix)]
        return out[: int(limit)]

    def _p_query_df(tickers: list[str], d0: pd.Timestamp, d1: pd.Timestamp) -> pd.DataFrame:
        """
        Pull only the rows needed for the portfolio simulator from DuckDB/S3.
        Returns columns: date, ticker, logret, close
        """
        if not tickers:
            return pd.DataFrame(columns=["date", "ticker", "logret", "close"])

        # DuckDB parameter binding does not accept a Python list directly as IN (...)
        # so we build the placeholders safely and pass params.
        placeholders = ", ".join(["?"] * len(tickers))
        sql = f"""
            SELECT
              date::DATE AS date,
              upper(ticker) AS ticker,
              logret,
              close
            FROM ae
            WHERE upper(ticker) IN ({placeholders})
              AND date::DATE BETWEEN ? AND ?
            ORDER BY date, ticker;
        """
        params = list(tickers) + [str(d0.date()), str(d1.date())]

        con = get_session_con()
        df = con.execute(sql, params).df()
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df["ticker"] = df["ticker"].astype(str).str.upper()
            df["logret"] = pd.to_numeric(df["logret"], errors="coerce")
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
        return df



    selected_p = reactive.Value([])
    selected_ind = reactive.Value([])

    @reactive.calc
    def ind_match_choices():
        q = input.ind_ticker_search()
        if not q or len(q.strip()) < 2:
            return []
        rows = _search_tickers(q, limit=30)
        return rows

    @reactive.effect
    def _update_ind_matches():
        choices = ind_match_choices()
        ui.update_selectize("ind_matches", choices=choices, selected=(choices[0] if choices else None))

    @reactive.effect
    @reactive.event(input.ind_add)
    def _ind_add_ticker():
        t = input.ind_matches()
        if not t:
            return
        cur = selected_ind.get()
        t = str(t).upper().strip()
        if t not in cur:
            selected_ind.set((cur + [t])[:10])

    @reactive.effect
    @reactive.event(input.ind_clear)
    def _ind_clear_tickers():
        selected_ind.set([])

    @output
    @render.text
    def ind_selected():
        cur = selected_ind.get()
        if not cur:
            return "Selected: (none)"
        return "Selected: " + ", ".join(cur)

    @reactive.calc
    def p_match_choices():
        q = input.p_ticker_search()
        if not q or len(q.strip()) < 2:
            return []
        rows = _search_tickers(q, limit=30)
        return rows

    @reactive.effect
    def _update_p_matches():
        choices = p_match_choices()
        ui.update_selectize("p_matches", choices=choices, selected=(choices[0] if choices else None))





    @reactive.effect
    @reactive.event(input.p_add)
    def _p_add_ticker():
        t = input.p_matches()
        if not t:
            return
        cur = selected_p.get()
        t = str(t).upper().strip()
        if t not in cur:
            selected_p.set((cur + [t])[:10])   # keep 1–10 like your UI

    @reactive.effect
    @reactive.event(input.p_clear)
    def _p_clear_tickers():
        selected_p.set([])

    @output
    @render.text
    def p_selected():
        cur = selected_p.get()
        if not cur:
            return "Selected: (none)"
        return "Selected: " + ", ".join(cur)




    @reactive.calc
    @reactive.event(input.p_go)
    def _p_params():
        tickers = selected_p.get()[:10]
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

        # Pull from DuckDB (instead of slicing pandas AE)
        df = _p_query_df(tickers, d0, d1)
        if df.empty:
            return pd.DataFrame()

        # daily simple returns
        df["r"] = _simple_returns(df["logret"])

        # Pivot to wide: one column per ticker
        r_wide = (
            df.pivot_table(index="date", columns="ticker", values="r", aggfunc="mean")
              .sort_index()
              .fillna(0.0)
        )
        if r_wide.shape[1] == 0:
            return pd.DataFrame()

        # Per-ticker cumulative index (rebased to start at 1)
        cum = (1.0 + r_wide).cumprod()
        cum = cum.div(cum.iloc[0].replace(0, np.nan), axis=1).fillna(1.0)

        if eq:
            # Equal-weight buy-and-hold
            n = r_wide.shape[1]
            w0 = pd.Series(1.0 / max(1, n), index=r_wide.columns)
            wealth_index = (cum.mul(w0, axis=1)).sum(axis=1)
        else:
            # Inverse-price static weights
            first_prices = (
                df.sort_values("date")
                  .dropna(subset=["close"])
                  .groupby("ticker")["close"].first()
                  .reindex(r_wide.columns)
            )

            invp = 1.0 / first_prices.replace(0, np.nan)
            invp = invp / invp.sum()

            if not np.isfinite(invp).any():
                invp = pd.Series(1.0 / r_wide.shape[1], index=r_wide.columns)

            invp = invp.fillna(0.0)
            wealth_index = (cum.mul(invp, axis=1)).sum(axis=1)

        wealth = wealth_index * amt
        r_daily = wealth.pct_change().fillna(0.0)
        r_total = wealth / wealth.iloc[0] - 1

        out = (
            pd.concat(
                [
                    wealth.rename("portfolio_$"),
                    r_daily.rename("portfolio_r_daily"),
                    r_total.rename("portfolio_r_total"),
                ],
                axis=1,
            )
            .join(cum.add_prefix("Cumulative return: "), how="left")
        )
        out.index = pd.to_datetime(out.index)
        return out


    @reactive.calc
    @reactive.event(input.p_go)
    def _p_meta():
        """
        Returns dict of Series (index = ticker):
          w0, p0, spent, shares0, final_values
        """
        tickers, amt, eq, d0, d1 = _p_params()
        if not tickers:
            return None

        df = _p_query_df(tickers, d0, d1)
        if df.empty:
            return None

        df["r"] = _simple_returns(df["logret"])
        r_wide = (
            df.pivot_table(index="date", columns="ticker", values="r", aggfunc="mean")
              .sort_index()
              .fillna(0.0)
        )
        if r_wide.shape[1] == 0:
            return None

        cum = (1.0 + r_wide).cumprod()
        cum = cum.div(cum.iloc[0].replace(0, np.nan), axis=1).fillna(1.0)

        p0 = (
            df.sort_values("date")
              .dropna(subset=["close"])
              .groupby("ticker")["close"].first()
              .reindex(r_wide.columns)
        )

        if eq:
            n = r_wide.shape[1]
            w0 = pd.Series(1.0 / max(1, n), index=r_wide.columns)
        else:
            invp = 1.0 / p0.replace(0, np.nan)
            if np.isfinite(invp).sum() == 0:
                w0 = pd.Series(1.0 / r_wide.shape[1], index=r_wide.columns)
            else:
                w0 = (invp / invp.sum()).fillna(0.0)

        spent = (amt * w0).fillna(0.0)
        shares0 = (spent / p0.replace(0, np.nan)).fillna(0.0)

        cum_last = cum.iloc[-1]
        final_values = (spent * cum_last).fillna(0.0)

        idx = r_wide.columns
        return {
            "w0": w0.loc[idx],
            "p0": p0.loc[idx],
            "spent": spent.loc[idx],
            "shares0": shares0.loc[idx],
            "final_values": final_values.loc[idx],
        }


    @reactive.calc
    def _p_color_map():
        m = _p_meta()
        if not m:
            return {}
        labels = (
            pd.Index(m["spent"].index)
              .union(m["shares0"].index)
              .union(m["final_values"].index)
        )
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
            unit="currency", percent=False,
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
        s = m["shares0"]
        return _pie_fig(
            names=s.index, values=s.values,
            title="Initial shares",
            unit="shares", percent=True,
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
        s = m["final_values"]
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
        end_val = float(df["portfolio_$"].iloc[-1])
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
                showarrow=False, font=dict(size=14),
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
            title_x=0.5,
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
        last = (
            df.tail(1)
              .drop(columns=["portfolio_r_daily", "portfolio_r_total", "portfolio_$"])
              .T.reset_index()
        )
        last.columns = ["Individual stock contributions", "Cumulative return index (start = 1)"]
        return last


    @output
    @render.download(filename=lambda: "portfolio_series.csv")
    @reactive.event(input.p_go)
    def p_dl():
        df = p_panel()
        yield df.reset_index().rename(columns={"index": "date"}).to_csv(index=False).encode()




    # ─────────────────────────────────────────────────────────────────────────
    # 2) SECTOR EXPLORER (DuckDB-backed)
    # ─────────────────────────────────────────────────────────────────────────

    def _s_query_df(secs: list[str], d0: pd.Timestamp, d1: pd.Timestamp) -> pd.DataFrame:
        """
        Pull only the rows needed for the sector explorer from DuckDB/S3.
        Returns: date, ticker, sector, logret, is_split_day, is_reverse_split_day
        """
        if not secs:
            return pd.DataFrame(columns=["date", "ticker", "sector", "logret", "is_split_day", "is_reverse_split_day"])

        # Normalize sectors to strings (DuckDB filter will match exact values)
        secs = [str(s).strip() for s in secs if str(s).strip()]
        if not secs:
            return pd.DataFrame(columns=["date", "ticker", "sector", "logret", "is_split_day", "is_reverse_split_day"])

        placeholders = ", ".join(["?"] * len(secs))
        sql = f"""
            SELECT
              date::DATE AS date,
              UPPER(ticker) AS ticker,
              TRIM(sector) AS sector,
              logret,
              COALESCE(is_split_day, false) AS is_split_day,
              COALESCE(is_reverse_split_day, false) AS is_reverse_split_day
            FROM ae
            WHERE TRIM(sector) IN ({placeholders})
              AND date::DATE BETWEEN ? AND ?
            ORDER BY date, ticker;
        """
        params = list(secs) + [str(d0.date()), str(d1.date())]

        con = get_session_con()
        df = con.execute(sql, params).df()
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df["ticker"] = df["ticker"].astype(str).str.upper()
            df["sector"] = df["sector"].astype("object")
            df["logret"] = pd.to_numeric(df["logret"], errors="coerce")
            # keep flags as bool
            df["is_split_day"] = df["is_split_day"].astype(bool)
            df["is_reverse_split_day"] = df["is_reverse_split_day"].astype(bool)
        return df


    @reactive.calc
    @reactive.event(input.s_go)
    def _s_params():
        secs = input.s_sectors() or []
        secs = [str(s).strip() for s in secs if str(s).strip()]
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

        # Pull from DuckDB (instead of slicing pandas AE)
        df = _s_query_df(secs, d0, d1)
        if df.empty:
            return pd.DataFrame()

        # Compute simple returns safely
        df["r"] = _simple_returns(df["logret"]).astype("float64")
        df["r"].replace([np.inf, -np.inf], np.nan, inplace=True)

        # Neutralize split days using flags from the same query (no join needed)
        split_hits = (df["is_split_day"] | df["is_reverse_split_day"]).to_numpy()
        df.loc[split_hits, "r"] = np.nan

        # De-dup (date,ticker) before sector averaging
        df = (
            df.groupby(["date", "ticker", "sector"], as_index=False, observed=True)["r"]
              .mean()
        )

        # Winsorize extremes
        df["r"] = df["r"].clip(lower=-0.95, upper=0.95)

        # Equal-weight within sector (your existing behavior)
        sec_daily = (
            df.groupby(["date", "sector"], observed=True)["r"]
              .mean()
              .unstack("sector")
              .sort_index()
              .fillna(0.0)
        )

        # Cumulative index per sector (start at 1)
        sec_cum = (1.0 + sec_daily.fillna(0.0)).cumprod()
        sec_cum = sec_cum.div(sec_cum.iloc[0].replace(0, np.nan), axis=1).fillna(1.0)

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
                showarrow=False, font=dict(size=14),
            )
            fig.update_layout(template="plotly_white", title_x=0.5)
            return fig

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
        last["Cumulative return index (start = 1)"] = last["Cumulative return index (start = 1)"].round(3)
        last["Total return"] = (last["Total return"] * 100).round(2).astype(str) + "%"
        return last


    @output
    @render.download(filename=lambda: "sector_indices.csv")
    @reactive.event(input.s_go)
    def s_dl():
        df = s_panel()
        yield df.reset_index().rename(columns={"index": "date"}).to_csv(index=False).encode()





    # ─────────────────────────────────────────────────────────────────────────
    # 3A) EVENT STUDY (DuckDB-backed)
    # ─────────────────────────────────────────────────────────────────────────

    def _es_query_panel(
        sel_types: list[str],
        k: int,
        date_lo: pd.Timestamp,
        date_hi: pd.Timestamp,
        sectors: list[str],
        no_overlap: bool,
    ) -> pd.DataFrame:
        sel_types = [str(x).strip() for x in (sel_types or []) if str(x).strip()]
        sectors   = [str(x).strip() for x in (sectors or []) if str(x).strip()]
        k = int(k)

        where_clauses = ["e.date BETWEEN ?::DATE AND ?::DATE"]
        params: list = [str(date_lo.date()), str(date_hi.date())]

        if sel_types:
            ph = ", ".join(["?"] * len(sel_types))
            where_clauses.append(f"e.event_type IN ({ph})")
            params.extend(sel_types)

        if sectors:
            ph = ", ".join(["?"] * len(sectors))
            where_clauses.append(f"e.sector IN ({ph})")
            params.extend(sectors)

        if no_overlap:
            where_clauses.append("e.is_overlap = FALSE")

        where_sql = " AND ".join(where_clauses)

        k = int(k)

        sql = f"""
        WITH ae_base AS (
          SELECT
            upper(ticker) AS ticker,
            date::DATE    AS date,
            logret,
            sector,
            tidx::INTEGER AS tidx
          FROM ae
          WHERE tidx IS NOT NULL
        ),
        events_scoped AS (
          SELECT
            upper(ticker) AS ticker,
            date::DATE    AS event_date,
            event_type,
            sector,
            n_events,
            is_overlap
          FROM events e
          WHERE {where_sql}
        ),
        events_with_tidx AS (
          SELECT es.*, a.tidx AS event_tidx
          FROM events_scoped es
          JOIN ae_base a
            ON a.ticker = es.ticker AND a.date = es.event_date
        ),
        offsets AS (
          SELECT * EXCLUDE(range), range AS rel_day
          FROM range(-{k}, {k} + 1)
        ),
        windows AS (
          SELECT
            e.ticker,
            e.event_type,
            e.sector,
            e.event_date,
            e.n_events,
            e.is_overlap,
            o.rel_day,
            (e.event_tidx + o.rel_day) AS tidx
          FROM events_with_tidx e
          CROSS JOIN offsets o
        )
        SELECT
          w.ticker,
          w.event_type,
          w.sector,
          w.event_date,
          a.date,
          w.rel_day,
          a.logret,
          w.n_events,
          w.is_overlap
        FROM windows w
        JOIN ae_base a
          ON a.ticker = w.ticker AND a.tidx = w.tidx
        ORDER BY w.event_date, w.ticker, a.date;
        """

        con = get_session_con()
        df = con.execute(sql, params).df()
        if df.empty:
            return pd.DataFrame()

        df["event_date"] = pd.to_datetime(df["event_date"]).dt.normalize()
        df["date"]       = pd.to_datetime(df["date"]).dt.normalize()
        df["rel_day"]    = pd.to_numeric(df["rel_day"], errors="coerce").astype("int16")
        df["ticker"]     = df["ticker"].astype(str).str.upper()
        df["event_type"] = df["event_type"].astype("object")
        df["logret"]     = pd.to_numeric(df["logret"], errors="coerce").astype("float64")
        df["sector"]     = df["sector"].astype("object")
        return df

    @reactive.calc
    @reactive.event(input.go)
    def _params():
        sel_types = input.etype() or []
        k = int(input.k())
        dr = input.dater()
        date_lo = pd.to_datetime(dr[0]).normalize() if dr else pd.Timestamp(dlo)
        date_hi = pd.to_datetime(dr[1]).normalize() if dr else pd.Timestamp(dhi)
        sectors = input.sector() or []
        no_overlap = bool(input.no_overlap())
        return sel_types, k, date_lo, date_hi, sectors, no_overlap

    @reactive.calc
    @reactive.event(input.go)
    def panel() -> pd.DataFrame:
        sel_types, k, date_lo, date_hi, sectors, no_overlap = _params()
        return _es_query_panel(sel_types, k, date_lo, date_hi, sectors, no_overlap)


    @output
    @render.text
    @reactive.event(input.go)
    def summary():
        df = panel()
        if df.empty:
            return "No events found. • Source: S3 (DuckDB)"
        n_ev = df[["ticker", "event_date", "event_type"]].drop_duplicates().shape[0]
        by_rel = df["rel_day"].value_counts().sort_index()
        return (
            f"Events matched: {n_ev} | Rows in ±k: {len(df):,} • "
            f"Min/Max rows per rel_day: {int(by_rel.min())}/{int(by_rel.max())} • "
            f"Source: S3 (DuckDB)"
        )


    @output
    @render.table
    @reactive.event(input.go)
    def tbl():
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
        mu = g.mean().sort_index()
        n  = g.count().sort_index()
        sd = g.std(ddof=1).sort_index()

        # standard error of mean per rel_day
        se = (sd / np.sqrt(n.clip(lower=1))).fillna(0.0)

        # CAR is cumulative sum of mean abnormal log returns
        car = mu.cumsum()

        # Propagate uncertainty across days (simple independence assumption)
        car_se = np.sqrt((se ** 2).cumsum())
        car_lo = car - 1.96 * car_se
        car_hi = car + 1.96 * car_se

        xvals = car.index.astype(int)

        # CI ribbon
        ci_rgba = "rgba(31, 119, 180, 0.20)"

        fig.add_trace(go.Scatter(
            x=xvals, y=car_hi.values,
            line=dict(width=0, color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False, name="CI hi"
        ))
        fig.add_trace(go.Scatter(
            x=xvals, y=car_lo.values,
            fill="tonexty",
            fillcolor=ci_rgba,
            line=dict(width=0, color="rgba(0,0,0,0)"),
            name="95% CI", showlegend=True, hoverinfo="skip"
        ))

        fig.add_trace(go.Scatter(
            x=xvals, y=car.values, mode="lines", name="Avg CAR",
            customdata=np.expm1(car.values),
            meta=n.values,
            hovertemplate=(
                "Day %{x}<br>"
                "CAR (log): %{y:.4f}<br>"
                "CAR (≈%): %{customdata:.2%}<br>"
                "n: %{meta}<extra></extra>"
            )
        ))

        fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.6)

        fig.update_layout(
            title="Average Cumulative Abnormal Log Return (±95% CI)",
            xaxis_title="Relative day",
            yaxis_title="Avg CAR (log units)",
            title_x=0.5,
            template="plotly_white",
            hovermode="x unified",
            margin=dict(t=60, r=20, b=40, l=60),
            legend=dict(itemsizing="constant"),
        )
        return fig


    @output
    @render.download(filename=lambda: "event_windows.csv")
    @reactive.event(input.go)
    def dl():
        df = panel()
        if df.empty:
            yield b""
            return
        yield df.to_csv(index=False).encode()


    # ─────────────────────────────────────────────────────────────────────────
    # 3B) EVENT STUDY — INDIVIDUAL STOCKS (DuckDB-backed)
    # ─────────────────────────────────────────────────────────────────────────

    def _ind_es_query_panel(
        sel_types: list[str],
        k: int,
        date_lo: pd.Timestamp,
        date_hi: pd.Timestamp,
        tickers: list[str],
    ) -> pd.DataFrame:
        sel_types = [str(x).strip() for x in (sel_types or []) if str(x).strip()]
        tickers   = [str(x).strip().upper() for x in (tickers or []) if str(x).strip()]
        k = int(k)

        where_clauses = ["event_date BETWEEN ?::DATE AND ?::DATE"]
        params: list = [str(date_lo.date()), str(date_hi.date())]

        if sel_types:
            placeholders = ", ".join(["?"] * len(sel_types))
            where_clauses.append(f"event_type IN ({placeholders})")
            params.extend(sel_types)

        if tickers:
            placeholders = ", ".join(["?"] * len(tickers))
            where_clauses.append(f"ticker IN ({placeholders})")
            params.extend(tickers)

        where_sql = " AND ".join(where_clauses)

        k = int(k)

        sql = f"""
        WITH ae_base AS (
          SELECT
            upper(ticker) AS ticker,
            date::DATE    AS date,
            logret,
            filing_form,
            COALESCE(is_split_day, false)         AS is_split_day,
            COALESCE(is_reverse_split_day, false) AS is_reverse_split_day,
            tidx::INTEGER AS tidx
          FROM ae
          WHERE tidx IS NOT NULL
        ),
        events_all AS (
          SELECT DISTINCT UPPER(ticker) AS ticker, date::DATE AS event_date, filing_form AS event_type
          FROM ae_base
          WHERE filing_form IS NOT NULL
          UNION ALL
          SELECT DISTINCT UPPER(ticker) AS ticker, date AS event_date, 'SPLIT' AS event_type
          FROM ae_base
          WHERE is_split_day = TRUE
          UNION ALL
          SELECT DISTINCT UPPER(ticker) AS ticker, date AS event_date, 'REVERSE_SPLIT' AS event_type
          FROM ae_base
          WHERE is_reverse_split_day = TRUE
        ),
        events_scoped AS (
          SELECT DISTINCT ticker, event_date, event_type
          FROM events_all
          WHERE {where_sql}
        ),
        events_with_tidx AS (
          SELECT es.*, a.tidx AS event_tidx
          FROM events_scoped es
          JOIN ae_base a
            ON a.ticker = es.ticker AND a.date = es.event_date
        ),
        offsets AS (
          SELECT * EXCLUDE(range), range AS rel_day
          FROM range(-{k}, {k} + 1)
        ),
        windows AS (
          SELECT
            e.ticker,
            e.event_type,
            e.event_date,
            o.rel_day,
            (e.event_tidx + o.rel_day) AS tidx
          FROM events_with_tidx e
          CROSS JOIN offsets o
        )
        SELECT
          w.ticker,
          w.event_type,
          w.event_date,
          a.date,
          w.rel_day,
          a.logret
        FROM windows w
        JOIN ae_base a
          ON a.ticker = w.ticker AND a.tidx = w.tidx
        ORDER BY w.ticker, w.event_date, a.date;
        """
        
        con = get_session_con()
        df = con.execute(sql, params).df()
        if df.empty:
            return pd.DataFrame()

        df["event_date"] = pd.to_datetime(df["event_date"]).dt.normalize()
        df["date"]       = pd.to_datetime(df["date"]).dt.normalize()
        df["rel_day"]    = pd.to_numeric(df["rel_day"], errors="coerce").astype("int16")
        df["ticker"]     = df["ticker"].astype(str).str.upper()
        df["event_type"] = df["event_type"].astype("object")
        df["logret"]     = pd.to_numeric(df["logret"], errors="coerce").astype("float64")
        return df


    @reactive.calc
    @reactive.event(input.ind_go)
    def _ind_params():
        sel_types = input.ind_etype() or []
        k = int(input.ind_k())
        dr = input.ind_dater()
        date_lo = pd.to_datetime(dr[0]).normalize() if dr else pd.Timestamp(dlo)
        date_hi = pd.to_datetime(dr[1]).normalize() if dr else pd.Timestamp(dhi)
        tickers = selected_ind.get()[:10]
        return sel_types, k, date_lo, date_hi, tickers


    @reactive.calc
    @reactive.event(input.ind_go)
    def ind_panel() -> pd.DataFrame:
        sel_types, k, date_lo, date_hi, tickers = _ind_params()
        return _ind_es_query_panel(sel_types, k, date_lo, date_hi, tickers)


    @output
    @render.text
    @reactive.event(input.ind_go)
    def ind_summary():
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

        g = df.groupby("rel_day")["logret"]
        mu = g.mean().sort_index()
        n  = g.count().sort_index()
        sd = g.std(ddof=1).sort_index()

        # standard error of mean per rel_day
        se = (sd / np.sqrt(n.clip(lower=1))).fillna(0.0)

        # CAR is cumulative sum of mean abnormal log returns
        car = mu.cumsum()

        # Propagate uncertainty across days (simple independence assumption)
        car_se = np.sqrt((se ** 2).cumsum())
        car_lo = car - 1.96 * car_se
        car_hi = car + 1.96 * car_se

        xvals = car.index.astype(int)

        ci_rgba = "rgba(31, 119, 180, 0.20)"

        fig.add_trace(go.Scatter(
            x=xvals, y=car_hi.values,
            line=dict(width=0, color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=xvals, y=car_lo.values,
            fill="tonexty",
            fillcolor=ci_rgba,
            line=dict(width=0, color="rgba(0,0,0,0)"),
            name="95% CI", showlegend=True, hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=xvals, y=car.values, mode="lines", name="Avg CAR",
            customdata=np.expm1(car.values),
            meta=n.values,
            hovertemplate=(
                "Day %{x}<br>"
                "CAR (log): %{y:.4f}<br>"
                "CAR (≈%): %{customdata:.2%}<br>"
                "n: %{meta}<extra></extra>"
            )
        ))

        fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.6)

        fig.update_layout(
            title="Average Cumulative Abnormal Log Return (±95% CI)",
            xaxis_title="Relative day",
            yaxis_title="Avg CAR (selected tickers)",
            title_x=0.5,
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
        df = ind_panel()
        if df.empty:
            yield b""
            return
        yield df.to_csv(index=False).encode()




    # ─────────────────────────────────────────────────────────────────────────
    # Appendix tables (cache CSV reads)
    # ─────────────────────────────────────────────────────────────────────────

    @output
    @render.data_frame
    def sec_desc_tbl():
        return render.DataTable(
            SEC_DESC_DF,
            filters=True,
            selection_mode="row",
            height="80vh",
            width="100%",
        )

    @output
    @render.data_frame
    def etfs_tbl():
        return render.DataTable(
            ETF_DF,
            filters=True,
            selection_mode="row",
            height="80vh",
            width="100%",
        )





# Bundle UI + server into a Shiny app
app = App(app_ui, server, static_assets=www_dir)
