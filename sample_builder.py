from pathlib import Path
import numpy as np
import pandas as pd

# optional: if pyarrow is available, use it to read schema fast
try:
    import pyarrow.parquet as pq
    _PA = True
except Exception:
    _PA = False

SRC = Path("outputs") / "analysis_enriched.parquet"
DST = Path("outputs") / "sample.parquet"

DATE_LO = pd.Timestamp("2024-09-30")
DATE_HI = pd.Timestamp("2025-09-30")

KEEP = [
    "date","ticker","close","logret","filing_form","sector",
    "is_split_day","is_reverse_split_day","market_cap"
]

# --- 1) discover available columns robustly (no rows read) ---
if _PA:
    cols_in_src = set(pq.ParquetFile(SRC).schema.names)
else:
    # fallback: read just the header by loading 1 small column (we’ll drop it)
    tmp = pd.read_parquet(SRC)  # last resort
    cols_in_src = set(tmp.columns)
    del tmp

use_cols = [c for c in KEEP if c in cols_in_src]
if not use_cols:
    raise RuntimeError("None of the requested columns exist in the source parquet.")

# --- 2) read the file with only those columns ---
df = pd.read_parquet(SRC, columns=use_cols, engine="pyarrow")

# --- 3) normalize + filter date ---
if "date" not in df.columns:
    raise RuntimeError("'date' column not found in the source. "
                       f"Available columns: {sorted(df.columns.tolist())}")

df["date"] = (
    pd.to_datetime(df["date"], errors="coerce")
      .dt.tz_localize(None)
      .dt.normalize()
)
df = df[df["date"].between(DATE_LO, DATE_HI)]

# --- 4) standardize keys ---
if "ticker" in df.columns:
    df["ticker"] = df["ticker"].astype(str).str.upper() 

# --- 5) ensure logret exists (compute from close if needed) ---
if "logret" not in df.columns and "close" in df.columns:
    df["close"] = pd.to_numeric(df["close"], errors="coerce").astype("float32")
    df["logret"] = (
        df.groupby("ticker", observed=True)["close"]
          .apply(lambda s: np.log(s / s.shift(1)))
          .reset_index(level=0, drop=True)
          .astype("float32")
    )
elif "logret" in df.columns:
    df["logret"] = pd.to_numeric(df["logret"], errors="coerce").astype("float32")

# --- 6) split flags present & boolean ---
for b in ("is_split_day","is_reverse_split_day"):
    if b not in df.columns:
        df[b] = False
    df[b] = df[b].fillna(False).astype("bool")

# --- 7) down-cast + categories ---
for f in ("close","market_cap"):
    if f in df.columns:
        df[f] = pd.to_numeric(df[f], errors="coerce").astype("float32")
for c in ("ticker","sector","filing_form"):
    if c in df.columns:
        df[c] = df[c].astype("category")

# --- 8) write ---
df = df.sort_values(["ticker","date"], kind="mergesort").reset_index(drop=True)
df.to_parquet(DST, index=False, engine="pyarrow")
print("Wrote:", DST, "rows:", len(df), "cols:", len(df.columns))