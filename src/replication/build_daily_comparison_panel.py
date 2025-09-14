import datetime as dt
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd


# ----------------------------
# Config
# ----------------------------
ROOT = Path(__file__).resolve().parents[2]  # .../ssrn-3984897-replication
DATA_DIR = ROOT / "data"

# Inputs:
DEX_V2_DAILY     = DATA_DIR / "raw" / "uniswap_v2" / "pair_day_data"
DEX_V3_DAILY     = DATA_DIR / "raw" / "uniswap_v3" / "pool_day_data"
DEX_PROXIES_DIR  = DATA_DIR / "interim" / "dex" / "proxies"
CEX_PROXIES_DIR  = DATA_DIR / "interim" / "cex" / "proxies"
GAS_FEES_DIR     = DATA_DIR / "raw" / "ethereum" / "gas_fees"
GAS_FEES_FILE    = GAS_FEES_DIR / "eth_gas_fees_daily.parquet"

# Output: processed comparison panel
OUT_DIR   = DATA_DIR / "processed" / "compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE        = OUT_DIR / "daily_panel.parquet"
OUT_FILE_V3AGG  = OUT_DIR / "daily_panel_agg.parquet"

# Replication window (final reporting)
TARGET_START = dt.date(2021, 3, 1)
TARGET_END   = dt.date(2023, 3, 1)

# Symbol → label map to align CEX with DEX labels
SYMB_TO_LABEL = {
    "ETH/USDC": "ETH-USDC",
    "ETH/USDT": "ETH-USDT",
    "BTC/ETH":  "ETH-BTC",
    "LINK/ETH": "LINK-ETH",
    "BTC/USDC": "BTC-USDC",
    "DAI/ETH":  "DAI-ETH",
    "MANA/ETH": "MANA-ETH",
    "USDC/USDT":"USDC-USDT",
    "DAI/USDT": "DAI-USDT",
    "AAVE/ETH": "AAVE-ETH",
    "BAT/ETH":  "BAT-ETH",
    "BTC/DAI":  "BTC-DAI",
    "CRV/ETH":  "CRV-ETH",
    "GRT/ETH":  "GRT-ETH",
    "KNC/ETH":  "KNC-ETH",
    "REP/ETH":  "REP-ETH",
    "SNX/ETH":  "SNX-ETH",
    "STORJ/ETH":"STORJ-ETH",
    "UNI/ETH":  "UNI-ETH",
    "OMG/ETH":  "OMG-ETH",
}


# ----------------------------
# Helpers
# ----------------------------
def _to_date_col(df: pd.DataFrame) -> pd.Series:
    """
    Standardize to a daily key (python date).
    Accepts any of: 'date' (ts-like), 'dayStartUnix' (seconds), 'timeSec' (seconds).
    """
    if "date" in df.columns:
        return pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert("UTC").dt.date
    if "dayStartUnix" in df.columns:
        return pd.to_datetime(pd.to_numeric(df["dayStartUnix"], errors="coerce"), unit="s", utc=True).dt.tz_convert("UTC").dt.date
    if "timeSec" in df.columns:
        return pd.to_datetime(pd.to_numeric(df["timeSec"], errors="coerce"), unit="s", utc=True).dt.tz_convert("UTC").dt.date
    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]").dt.date


def _valid_frame(df: pd.DataFrame) -> bool:
    """True if df has rows and at least one column that is not entirely NA."""
    return (
        isinstance(df, pd.DataFrame)
        and df.shape[0] > 0
        and any(~df[c].isna().all() for c in df.columns)
    )


def _drop_all_na_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that are entirely NA (prevents concat FutureWarning)."""
    return df.loc[:, ~df.isna().all()]


# ----------------------------
# Loaders (order matters)
# ----------------------------
def _load_dex_daily(labels: Optional[List[str]] = None) -> pd.DataFrame:
    """Read Uniswap v2 & v3 daily aggregates into a tidy frame."""
    rows = []

    # v2
    for f in DEX_V2_DAILY.glob("*.parquet"):
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        if df.empty:
            continue
        lbl = f.stem.replace("_", "-")
        if labels is not None and lbl not in labels:
            continue
        d = df.copy()
        d["date"] = _to_date_col(d)
        d["venue_type"] = "DEX"
        d["venue"] = "uniswap_v2"
        d["exchange"] = "uniswap_v2"
        d["label"] = lbl
        rows.append(d)

    # v3
    for f in DEX_V3_DAILY.glob("*.parquet"):
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        if df.empty:
            continue

        base = f.stem
        if "_fee" in base:
            lbl = base.split("_fee")[0].replace("_", "-")
            fee = base.split("_fee")[1]
            venue_name = f"uniswap_v3_fee{fee}"
        else:
            lbl = base.replace("_", "-")
            venue_name = "uniswap_v3"

        if labels is not None and lbl not in labels:
            continue

        d = df.copy()
        d["date"] = _to_date_col(d)
        d["venue_type"] = "DEX"
        d["venue"] = venue_name
        d["exchange"] = venue_name
        d["label"] = lbl
        rows.append(d)

    rows = [_drop_all_na_columns(r) for r in rows if _valid_frame(r)]
    if not rows:
        return pd.DataFrame()

    keep_cols = ["date", "venue_type", "venue", "exchange", "label",
                 "close", "high", "low", "ret", "volumeUSD"]
    out = pd.concat(rows, ignore_index=True)
    for c in keep_cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[keep_cols]
    return out


def _load_dex_proxies(labels: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Read DEX proxy parquet files (built from daily) and return
    only proxy columns keyed by (label, venue, date) for merging onto DEX daily.
    """
    rows = []
    exdirs = [p for p in DEX_PROXIES_DIR.glob("*") if p.is_dir()]
    for exdir in exdirs:
        venue = exdir.name  # e.g., uniswap_v2, uniswap_v3_fee3000
        for f in exdir.glob("*_proxies.parquet"):
            try:
                df = pd.read_parquet(f)
            except Exception:
                continue
            if df.empty:
                continue
            stem = f.stem  # like ETH-USDC_proxies
            label = stem.replace("_proxies", "")
            if labels is not None and label not in labels:
                continue

            d = df.copy()
            d["date"] = _to_date_col(d)
            d["label"] = label
            d["venue"] = venue
            # Keep only proxy columns for merge
            keep = ["label", "venue", "date", "proxy_chl", "proxy_cs", "proxy_amihud", "proxy_roll"]
            for c in keep:
                if c not in d.columns:
                    d[c] = np.nan
            rows.append(d[keep])

    if not rows:
        return pd.DataFrame()
    out = pd.concat([_drop_all_na_columns(r) for r in rows if _valid_frame(r)], ignore_index=True)
    # Deduplicate just in case
    out = (out.sort_values(["label", "venue", "date"])
              .drop_duplicates(subset=["label", "venue", "date"], keep="last")
              .reset_index(drop=True))
    return out


def _load_cex_proxies(exchanges: Optional[List[str]] = None,
                      labels: Optional[List[str]] = None) -> pd.DataFrame:
    """Read all CEX proxy parquet files into a tidy frame."""
    rows = []
    exdirs = [p for p in CEX_PROXIES_DIR.glob("*") if p.is_dir()]
    if exchanges:
        exdirs = [CEX_PROXIES_DIR / e for e in exchanges if (CEX_PROXIES_DIR / e).is_dir()]

    for exdir in exdirs:
        exchange = exdir.name
        for f in exdir.glob("*_proxies.parquet"):
            try:
                df = pd.read_parquet(f)
            except Exception:
                continue
            if df.empty:
                continue

            symbol = (str(df["symbol"].iloc[0]) if "symbol" in df.columns and not df["symbol"].isna().all()
                      else f.stem.replace("_proxies", "").replace("-", "/"))
            label = SYMB_TO_LABEL.get(symbol)
            if label is None:
                continue
            if labels is not None and label not in labels:
                continue

            d = df.copy()
            d["date"] = _to_date_col(d)
            d["venue_type"] = "CEX"
            d["venue"] = exchange
            d["exchange"] = exchange
            d["label"] = label
            d["symbol"] = symbol
            rows.append(d)

    rows = [_drop_all_na_columns(r) for r in rows if _valid_frame(r)]
    if not rows:
        return pd.DataFrame()

    keep_cols = ["date", "venue_type", "venue", "exchange", "label", "symbol",
                 "close", "high", "low", "ret", "volume_quote",
                 "proxy_chl", "proxy_cs", "proxy_amihud", "proxy_roll"]
    out = pd.concat(rows, ignore_index=True)
    for c in keep_cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[keep_cols]
    return out


def _load_eth_gas_fees() -> pd.DataFrame:
    """
    Load daily ETH gas fees (from Dune export) and standardize to a 'date' key.
    Source: data/raw/ethereum/gas_fees/eth_gas_fees_daily.parquet
    """
    p = GAS_FEES_FILE
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(p)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()

    d = df.copy()
    d["date"] = pd.to_datetime(d["day"], utc=True, errors="coerce").dt.tz_convert("UTC").dt.date

    rename = {
        "median_base_fee_gwei":            "eth_median_base_fee_gwei",
        "median_priority_fee_gwei":        "eth_median_priority_fee_gwei",
        "median_effective_gas_price_gwei": "eth_median_effective_gas_price_gwei",
        "median_total_fee_gwei":           "eth_median_total_fee_gwei",
        "n_blocks_with_basefee":           "eth_n_blocks_with_basefee",
        "n_blocks":                        "eth_n_blocks",
        "n_txs":                           "eth_n_txs",
    }
    for c in rename.keys():
        if c not in d.columns:
            d[c] = np.nan

    d = d[["date"] + list(rename.keys())].rename(columns=rename)
    d = d.sort_values("date").drop_duplicates("date", keep="last")
    return d


# ----------------------------
# Orchestrator
# ----------------------------
def run_build_daily_comparison_panel(exchanges: Optional[List[str]] = None,
                                     labels: Optional[List[str]] = None,
                                     start_date: Optional[dt.date] = TARGET_START,
                                     end_date: Optional[dt.date] = TARGET_END) -> None:
    """
    Build a tidy daily panel combining:
      - DEX daily aggregates (v2 & v3) from raw subgraph outputs
      - DEX proxies (merged onto DEX rows)
      - CEX proxies (CHL/CS/Amihud/Roll) from interim
      - ETH gas-fee metrics (daily) from raw exports
    Outputs:
      - data/processed/compare/daily_panel.parquet        (detailed, all v3 fee tiers)
      - data/processed/compare/daily_panel_agg.parquet    (adds v3 fee–tier aggregate as 'uniswap_v3_all')
    """
    # 1) DEX daily
    dex = _load_dex_daily(labels=labels)

    # 2) DEX proxies → merge onto DEX daily
    dex_px = _load_dex_proxies(labels=labels)
    if not dex.empty and not dex_px.empty:
        dex = dex.merge(dex_px, on=["label", "venue", "date"], how="left")
    elif dex.empty and not dex_px.empty:
        # Fallback: build a minimal DEX frame from proxies if daily is missing
        d = dex_px.copy()
        d["venue_type"] = "DEX"
        d["exchange"] = d["venue"]
        for col in ["close", "high", "low", "ret", "volumeUSD"]:
            d[col] = np.nan
        cols = ["date", "venue_type", "venue", "exchange", "label",
                "close", "high", "low", "ret", "volumeUSD",
                "proxy_chl", "proxy_cs", "proxy_amihud", "proxy_roll"]
        dex = d[cols]

    # 3) CEX proxies
    cex = _load_cex_proxies(exchanges=exchanges, labels=labels)

    frames = []
    if not dex.empty:
        frames.append(dex)
    if not cex.empty:
        c = cex.rename(columns={"volume_quote": "volumeUSD"}).copy()
        frames.append(c)

    if not frames:
        print("[INFO] nothing to write; no inputs found.")
        return

    panel = pd.concat(frames, ignore_index=True)

    # Trim to reporting window
    if start_date is not None:
        panel = panel.loc[panel["date"] >= start_date]
    if end_date is not None:
        panel = panel.loc[panel["date"] < end_date]

    # 4) ETH gas fees (by date)
    fees = _load_eth_gas_fees()
    if not fees.empty:
        panel = panel.merge(fees, on="date", how="left")

    # Stable sort and save the detailed panel
    panel = panel.sort_values(["label", "date", "venue_type", "venue"]).reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_FILE, index=False)
    print(f"[OK] daily comparison panel -> {OUT_FILE}")

    # ---- v3 fee–tier aggregation (adds a synthetic 'uniswap_v3_all' row per label × day) ----
    v3_mask = panel["venue"].str.startswith("uniswap_v3_fee", na=False)
    v3 = panel.loc[v3_mask, ["label", "date", "volumeUSD"]].copy()
    if not v3.empty:
        v3_agg = (v3.groupby(["label", "date"], as_index=False)
                    .agg(volumeUSD=("volumeUSD", "sum")))
        v3_agg["venue_type"] = "DEX"
        v3_agg["venue"] = "uniswap_v3_all"
        v3_agg["exchange"] = "uniswap_v3_all"

        # Ensure all panel columns exist on v3_agg; fill non-volume fields with NaN
        for col in panel.columns:
            if col not in v3_agg.columns:
                v3_agg[col] = np.nan

        # Order columns to match panel
        v3_agg = v3_agg[panel.columns]

        # Avoid FutureWarning: drop all-NA columns before concat
        v3_agg_nz = v3_agg.loc[:, ~v3_agg.isna().all()]
        panel_agg = pd.concat([panel, v3_agg_nz], ignore_index=True)
        panel_agg = panel_agg.reindex(columns=panel.columns)

        panel_agg = panel_agg.sort_values(["label", "date", "venue_type", "venue"]).reset_index(drop=True)
        panel_agg.to_parquet(OUT_FILE_V3AGG, index=False)
        print(f"[OK] daily comparison panel with v3 aggregation -> {OUT_FILE_V3AGG}")
    else:
        panel.to_parquet(OUT_FILE_V3AGG, index=False)
        print(f"[INFO] no v3 fee-tier rows found; wrote detailed panel as agg -> {OUT_FILE_V3AGG}")


if __name__ == "__main__":
    run_build_daily_comparison_panel()
