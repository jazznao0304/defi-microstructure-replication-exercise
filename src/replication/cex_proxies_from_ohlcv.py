from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path

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
RAW_OHLCV    = DATA_DIR / "raw" / "cex" / "ohlcv"
INTERIM_DIR  = DATA_DIR / "interim" / "cex" / "proxies"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

# Rolling windows (trading days) for low-frequency spread estimators
ROLL_WINDOW = 21  # Used for Roll and CHL expectations


# ----------------------------
# Proxy helpers (vectorized)
# ----------------------------
def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize OHLCV to a common daily schema and add basic fields.
    Expects columns: timestamp (ms), open, high, low, close, volume, exchange, symbol, timeframe
    """
    d = df.copy()
    # Deterministic ordering before any rolling/window ops
    if "timestamp" in d.columns:
        d = d.sort_values("timestamp").reset_index(drop=True)

    # Daily key aligned with DEX datasets: UTC day start in epoch seconds
    d["dayStartUnix"] = ((d["timestamp"] // 1000) // 86400) * 86400
    # Optional human-readable date (UTC)
    d["date"] = pd.to_datetime(d["dayStartUnix"], unit="s", utc=True)

    # Close-to-close log return
    d["ret"] = np.log(d["close"]).diff()

    # Quote (dollar) volume proxy: base_volume * mid (HL2)
    d["mid_hl2"] = (d["high"] + d["low"]) / 2.0
    d["vol_quote"] = d["volume"] * d["mid_hl2"]
    return d


def _roll_spread(d: pd.DataFrame, window: int = ROLL_WINDOW) -> pd.Series:
    """Roll (1984): 2 * sqrt( -Cov(ΔP_t, ΔP_{t-1}) ) using close-to-close returns."""
    x = d["ret"]
    # Only compute when the full window has no NaNs
    cov = x.rolling(window).apply(
        lambda s: np.cov(s[1:], s[:-1], ddof=0)[0, 1] if s.isna().sum() == 0 and len(s) > 1 else np.nan,
        raw=False,
    )
    spread = 2.0 * np.sqrt(np.maximum(-cov, 0.0))
    spread[cov >= 0] = np.nan
    return spread


def _amihud(d: pd.DataFrame) -> pd.Series:
    """Amihud (2002) illiquidity: |return| / dollar_volume."""
    x = d.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        illiq = np.abs(x["ret"]) / x["vol_quote"]
    return illiq.replace([np.inf, -np.inf], np.nan)


def _corwin_schultz(d: pd.DataFrame) -> pd.Series:
    """
    Corwin–Schultz (2012) high–low spread estimator (two-day highs/lows).
    Returns an estimate of the effective spread in relative terms.
    """
    H = d["high"].astype(float)
    L = d["low"].astype(float)

    beta = (np.log(H / L)) ** 2 + (np.log(H.shift(1) / L.shift(1))) ** 2
    gamma = (np.log((H.combine(H.shift(1), np.maximum) / L.combine(L.shift(1), np.minimum)))) ** 2

    # k as in the paper
    denom = (3 - 2 ** 1.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = (np.sqrt(2 * beta) - np.sqrt(beta)) / denom

    # CS spread proxy
    with np.errstate(over="ignore", invalid="ignore"):
        cs = 2 * (np.exp(k) - 1) / (1 + np.exp(k))

    # Guard rails
    cs[(beta <= 0) | (gamma <= 0) | (~np.isfinite(cs))] = np.nan
    return cs


def _chl_spread(d: pd.DataFrame, window: int = ROLL_WINDOW) -> pd.Series:
    """CHL (Abdi & Ranaldo, 2017): 2 * sqrt( E[(c_t - m_t)(c_{t+1} - m_t)] )."""
    c = np.log(d["close"])
    m = np.log((d["high"] + d["low"]) / 2.0)
    x = (c - m)
    y = x.shift(-1)  # c_{t+1} - m_t
    cov_like = (x * y).rolling(window).mean()
    chl = 2.0 * np.sqrt(np.maximum(cov_like, 0.0))
    chl[cov_like <= 0] = np.nan
    return chl


# ----------------------------
# Orchestrator
# ----------------------------
@dataclass
class ProxyTask:
    exchanges: Optional[List[str]] = None   # None -> auto-discover under RAW_OHLCV
    symbols:   Optional[List[str]] = None   # None -> per-exchange auto
    timeframe: str = "1d"                   # Expects daily input for CHL/CS stability


def run_cex_proxies_from_ohlcv(task: ProxyTask = ProxyTask()) -> None:
    """
    Build daily proxy series (CHL (2017), CS (2012), Amihud (2002), Roll (1984)) from raw CEX OHLCV.
    Writes per-exchange parquet to: data/interim/cex/proxies/{exchange}/{symbol}_proxies.parquet
    """
    # Discover exchanges
    base = RAW_OHLCV
    if task.exchanges:
        exdirs = [base / e for e in task.exchanges]
    else:
        exdirs = [p for p in base.glob("*") if p.is_dir()]

    for exdir in exdirs:
        exchange = exdir.name
        out_dir = INTERIM_DIR / exchange
        out_dir.mkdir(parents=True, exist_ok=True)

        # Discover symbols for this exchange (by timeframe suffix)
        files = list(exdir.glob(f"*_{task.timeframe}.parquet"))
        if task.symbols:
            candidates = [exdir / f"{s.replace('/','-')}_{task.timeframe}.parquet" for s in task.symbols]
            files = [f for f in candidates if f.exists()]

        if not files:
            print(f"[INFO] no OHLCV files for {exchange} ({task.timeframe}).")
            continue

        for f in files:
            try:
                df = pd.read_parquet(f)
            except Exception as e:
                print(f"[WARN] read failed {f}: {e}")
                continue
            if df.empty:
                continue
            if "timeframe" in df.columns and str(df["timeframe"].iloc[0]) != str(task.timeframe):
                # Skip mismatched timeframe
                continue

            symbol = df["symbol"].iloc[0] if "symbol" in df.columns else f.name.split("_")[0].replace("-", "/")

            d = _prep(df)

            out = pd.DataFrame({
                "dayStartUnix": d["dayStartUnix"],
                "date": d["date"],
                "exchange": exchange,
                "symbol": symbol,
                # Proxies (fixed order for consistency)
                "proxy_chl": _chl_spread(d, window=ROLL_WINDOW),
                "proxy_cs": _corwin_schultz(d),
                "proxy_amihud": _amihud(d),
                "proxy_roll": _roll_spread(d, window=ROLL_WINDOW),
                # Reference fields
                "close": d["close"],
                "high": d["high"],
                "low": d["low"],
                "volume_base": d["volume"],
                "volume_quote": d["vol_quote"],
                "ret": d["ret"],
            })

            # Drop rows where all proxies are NaN (e.g., initial warmup windows)
            keep = (~out[["proxy_chl", "proxy_cs", "proxy_amihud", "proxy_roll"]].isna()).any(axis=1)
            out = out.loc[keep].reset_index(drop=True)

            # Deterministic order before saving (align with DEX daily)
            out = out.sort_values(["dayStartUnix", "exchange", "symbol"]).reset_index(drop=True)

            tag = f"{symbol.replace('/','-')}_proxies.parquet"
            outp = out_dir / tag
            out.to_parquet(outp, index=False)
            print(f"[OK] {exchange}:{symbol} -> {outp}")

if __name__ == "__main__":
    run_cex_proxies_from_ohlcv()
