from typing import List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
from dotenv import load_dotenv


# ----------------------------
# Config
# ----------------------------
ROOT = Path(__file__).resolve().parents[2]  # .../ssrn-3984897-replication
ENV_PATH = ROOT / "config" / ".env"
load_dotenv(dotenv_path=ENV_PATH)

DATA_DIR     = ROOT / "data"
RAW_V2_DIR   = DATA_DIR / "raw" / "uniswap_v2" / "pair_day_data"
RAW_V3_DIR   = DATA_DIR / "raw" / "uniswap_v3" / "pool_day_data"
INTERIM_DIR  = DATA_DIR / "interim" / "dex" / "proxies"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

# Rolling window (trading days) for low-frequency estimators (match CEX)
ROLL_WINDOW = 21


# ----------------------------
# Helpers (vectorized)
# ----------------------------
def _parse_v3_label_and_fee(stem: str) -> Tuple[str, Optional[int]]:
    """
    Input stem examples:
      - 'ETH-USDC_fee500' -> ('ETH-USDC', 500)
      - 'ETH-USDT'        -> ('ETH-USDT', None)
    """
    m = re.search(r"(.+)_fee(\d+)$", stem)
    if m:
        return (m.group(1).replace("_", "-"), int(m.group(2)))
    return (stem.replace("_", "-"), None)


def _prep_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize Uniswap v2 day-level data and add basic fields.
    Expects: 'dayStartUnix' (seconds), 'price_1_per_0_derived', 'volumeUSD'
    """
    d = df.copy()
    if "dayStartUnix" not in d.columns:
        raise ValueError("v2 daily requires 'dayStartUnix' column.")
    d = d.sort_values("dayStartUnix").reset_index(drop=True)

    # UTC date (human-readable)
    d["date"] = pd.to_datetime(d["dayStartUnix"], unit="s", utc=True)

    # Price & returns (guard against nonpositive)
    price = pd.to_numeric(d.get("price_1_per_0_derived"), errors="coerce").astype(float)
    price = price.where(price > 0)  # nonpositive -> NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.log(price)
    d["close"] = price
    d["ret"] = logp.diff()

    # Dollar volume
    d["volumeUSD"] = pd.to_numeric(d.get("volumeUSD"), errors="coerce")

    return d


def _prep_v3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize Uniswap v3 day-level data and add basic fields.
    Expects: 'dayStartUnix' (seconds), 'token0Price' (or a consistent price), 'volumeUSD'
    """
    d = df.copy()
    if "dayStartUnix" not in d.columns:
        raise ValueError("v3 daily requires 'dayStartUnix' column.")
    d = d.sort_values("dayStartUnix").reset_index(drop=True)

    d["date"] = pd.to_datetime(d["dayStartUnix"], unit="s", utc=True)

    # Use token0Price; guard against nonpositive
    price = pd.to_numeric(d.get("token0Price"), errors="coerce").astype(float)
    price = price.where(price > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.log(price)
    d["close"] = price
    d["ret"] = logp.diff()

    d["volumeUSD"] = pd.to_numeric(d.get("volumeUSD"), errors="coerce")

    return d


def _roll_spread(d: pd.DataFrame, window: int = ROLL_WINDOW) -> pd.Series:
    """Roll (1984): 2 * sqrt( -Cov(ΔP_t, ΔP_{t-1}) ) using close-to-close returns."""
    x = d["ret"]
    cov = x.rolling(window).apply(
        lambda s: np.cov(s[1:], s[:-1], ddof=0)[0, 1] if s.isna().sum() == 0 and len(s) > 1 else np.nan,
        raw=False,
    )
    spread = 2.0 * np.sqrt(np.maximum(-cov, 0.0))
    spread[cov >= 0] = np.nan
    return spread


def _amihud(d: pd.DataFrame) -> pd.Series:
    """Amihud (2002) illiquidity proxy: |return| / dollar_volume."""
    x = d.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        illiq = np.abs(x["ret"]) / x["volumeUSD"]
    return illiq.replace([np.inf, -np.inf], np.nan)


# ----------------------------
# Orchestrator
# ----------------------------
@dataclass
class DexProxyTask:
    venues: Optional[List[str]] = None      # None -> ["uniswap_v2", "uniswap_v3"]
    labels: Optional[List[str]] = None      # None -> process all labels available


def run_dex_proxies_from_daily(task: DexProxyTask = DexProxyTask()) -> None:
    """
    Build daily DEX proxy series (Amihud (2002), Roll (1984)) from Uniswap v2/v3 day-level data.
    Output per-venue parquet to:
        data/interim/dex/proxies/{venue}/{label}_proxies.parquet

    Notes:
      - CHL and Corwin–Schultz are not computed here (need daily high/low).
      - v3 files may represent distinct fee tiers; we keep them under distinct venue folders
        e.g., 'uniswap_v3_fee500'.
    """
    venues = task.venues or ["uniswap_v2", "uniswap_v3"]

    for venue in venues:
        if venue not in {"uniswap_v2", "uniswap_v3"}:
            print(f"[WARN] unknown venue '{venue}' – skipping.")
            continue

        src_dir = RAW_V2_DIR if venue == "uniswap_v2" else RAW_V3_DIR
        files = sorted(src_dir.glob("*.parquet"))
        if not files:
            print(f"[INFO] no daily files for {venue}.")
            continue

        for f in files:
            try:
                df = pd.read_parquet(f)
            except Exception as e:
                print(f"[WARN] read failed {f}: {e}")
                continue
            if df.empty:
                continue

            if venue == "uniswap_v2":
                label = f.stem.replace("_", "-")
                if task.labels is not None and label not in task.labels:
                    continue
                d = _prep_v2(df)
                venue_name = "uniswap_v2"
            else:
                # v3: parse label and fee tier
                label_base, fee = _parse_v3_label_and_fee(f.stem)
                if task.labels is not None and label_base not in task.labels:
                    continue
                d = _prep_v3(df)
                venue_name = f"uniswap_v3_fee{fee}" if fee is not None else "uniswap_v3"

            # Build output frame
            out = pd.DataFrame({
                "dayStartUnix": d["dayStartUnix"],
                "date": d["date"],
                "venue": venue_name,
                "label": label if venue == "uniswap_v2" else label_base,
                # Proxies available at daily frequency
                "proxy_amihud": _amihud(d),
                "proxy_roll": _roll_spread(d, window=ROLL_WINDOW),
                # Reference fields
                "close": d["close"],
                "volumeUSD": d["volumeUSD"],
                "ret": d["ret"],
            })

            # Drop rows where both proxies are NaN (e.g., warm-up, missing price/volume)
            keep = (~out[["proxy_amihud", "proxy_roll"]].isna()).any(axis=1)
            out = out.loc[keep].reset_index(drop=True)

            # Deterministic order before saving (align with CEX proxies)
            out = out.sort_values(["dayStartUnix", "venue", "label"]).reset_index(drop=True)

            # Save under per-venue folder; filename is label-based (like CEX uses symbol)
            out_dir = INTERIM_DIR / venue_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{(label if venue=='uniswap_v2' else label_base)}_proxies.parquet"
            out.to_parquet(out_path, index=False)
            print(f"[OK] {venue_name}:{(label if venue=='uniswap_v2' else label_base)} -> {out_path}")


if __name__ == "__main__":
    run_dex_proxies_from_daily()
