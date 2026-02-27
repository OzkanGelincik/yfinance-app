from pathlib import Path
import os
import duckdb
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
OUT_DIR = APP_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

S3_URI = os.getenv(
    "AE_S3_URI",
    "s3://yfinance-app-data/yfinance-app/analysis_enriched_backfilled_v7_3y_11col_tidx_year.parquet",
)

def configure(con):
    def safe(ext):
        try:
            con.execute(f"INSTALL {ext};")
        except Exception:
            pass
        con.execute(f"LOAD {ext};")
    safe("httpfs")
    safe("aws")
    con.execute("CALL load_aws_credentials();")

con = duckdb.connect(database=":memory:")
configure(con)

df = con.execute(f"""
    SELECT DISTINCT UPPER(ticker) AS ticker
    FROM read_parquet('{S3_URI}')
    WHERE ticker IS NOT NULL
    ORDER BY 1
""").df()

(df[["ticker"]]).to_csv(OUT_DIR / "tickers.csv", index=False)
print("Wrote:", OUT_DIR / "tickers.csv", "rows:", len(df))