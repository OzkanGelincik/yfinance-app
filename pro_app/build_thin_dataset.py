from pathlib import Path
import duckdb

APP_DIR = Path(__file__).resolve().parent
OUT_DIR = APP_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SRC = OUT_DIR / "analysis_enriched_backfilled_v7.parquet"
OUT = OUT_DIR / "analysis_enriched_backfilled_v7_3y_11col_tidx_year.parquet"

if not SRC.exists():
    raise FileNotFoundError(f"Source parquet not found: {SRC}")

src = SRC.as_posix()
out = OUT.as_posix()

con = duckdb.connect()

con.execute(f"""
COPY (
  SELECT
    CAST(date AS DATE) AS date,
    UPPER(ticker) AS ticker,

    ROW_NUMBER() OVER (
      PARTITION BY UPPER(ticker)
      ORDER BY CAST(date AS DATE)
    ) - 1 AS tidx,

    close,
    logret,
    filing_form,
    NULLIF(TRIM(sector), '') AS sector,

    -- robust bool casting: NULL -> FALSE, 0 -> FALSE, nonzero -> TRUE
    COALESCE(CAST(is_split_day != 0 AS BOOLEAN), FALSE) AS is_split_day,
    COALESCE(CAST(is_reverse_split_day != 0 AS BOOLEAN), FALSE) AS is_reverse_split_day,

    market_cap,
    EXTRACT(YEAR FROM CAST(date AS DATE))::INTEGER AS year

  FROM read_parquet('{src}')
) TO '{out}' (FORMAT PARQUET);
""")

con.close()
print(f"Wrote: {OUT}")