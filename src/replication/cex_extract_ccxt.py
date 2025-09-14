import time
import random
import datetime as dt
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import ccxt


# ----------------------------
# Config
# ----------------------------
ROOT = Path(__file__).resolve().parents[2]  # .../ssrn-3984897-replication
ENV_PATH = ROOT / "config" / ".env"
load_dotenv(dotenv_path=ENV_PATH)

DATA_DIR   = ROOT / "data"
RAW_DIR    = DATA_DIR / "raw" / "cex"
META_DIR   = RAW_DIR / "meta"
OHLCV_DIR  = RAW_DIR / "ohlcv"
TRADES_DIR = RAW_DIR / "trades"
OB_DIR     = RAW_DIR / "orderbooks"

for d in (META_DIR, OHLCV_DIR, TRADES_DIR, OB_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Default window (UTC) aligned with DEX scripts
START = int(dt.datetime(2021, 2, 1, tzinfo=dt.timezone.utc).timestamp())  # Buffer for warm-up
END   = int(dt.datetime(2023, 3, 1, tzinfo=dt.timezone.utc).timestamp())

# Symbols to consider (CCXT format). Adjust per exchange support.
SYMBOLS: List[str] = [
    "ETH/USDC", "ETH/USDT", "BTC/ETH", "LINK/ETH", "BTC/USDC",
    "DAI/ETH", "MANA/ETH", "USDC/USDT",
    "DAI/USDT", "AAVE/ETH", "BAT/ETH", "BTC/DAI",
    "CRV/ETH", "GRT/ETH", "KNC/ETH", "REP/ETH",
    "SNX/ETH", "STORJ/ETH", "UNI/ETH", "OMG/ETH",
]

# Popular CEX venues (names must match ccxt exchange ids)
EXCHANGES: List[str] = ["binance", "kraken", "coinbase", "okx", "huobi"]  # adjust as needed

DEFAULT_TIMEFRAME = "1d"  # Daily bars by default


# ----------------------------
# Helpers
# ----------------------------
def _make_exchange(eid: str, rate_limit: bool = True) -> ccxt.Exchange:
    cls = getattr(ccxt, eid)
    ex = cls({
        "enableRateLimit": rate_limit,
        "timeout": 30000,
    })
    ex.load_markets()
    return ex

def _ms(ts_sec: int) -> int:
    return ts_sec * 1000

def _sleep(ex: ccxt.Exchange):
    # Basic backoff respecting exchange rate limit (+ small jitter to avoid thundering herd)
    time.sleep(getattr(ex, "rateLimit", 1000) / 1000.0 + random.uniform(0, 0.25))

def _alias_symbols_for_exchange(ex_id: str, syms: List[str]) -> List[str]:
    # Minimal aliasing for Kraken which uses XBT instead of BTC
    if ex_id == "kraken":
        return [s.replace("BTC/", "XBT/") for s in syms]
    return syms


# ----------------------------
# Fetch OHLCV, trades, orderbook
# ----------------------------
def fetch_ohlcv_range(
    ex: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    start_ts: int,
    end_ts: int,
    limit: int = 1000,
) -> pd.DataFrame:
    """Paginate OHLCV from CCXT over [start_ts, end_ts)."""
    since = _ms(start_ts)
    out: List[List[float]] = []
    while True:
        try:
            batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        except ccxt.BaseError:
            _sleep(ex); continue
        if not batch:
            break
        out.extend(batch)
        last = batch[-1][0]
        if last >= _ms(end_ts) - 1:
            break
        since = last + 1
        _sleep(ex)

    if not out:
        return pd.DataFrame()

    df = pd.DataFrame(out, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["exchange"] = ex.id
    df["symbol"] = symbol
    df["timeframe"] = timeframe

    # Standardize time column to common schema (seconds since epoch)
    df["timeSec"] = (df["timestamp"] // 1000).astype("int64")

    # Ensure numeric dtypes and drop duplicate timestamps if any
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates(subset=["timeSec"]).sort_values("timeSec").reset_index(drop=True)

    # Clip strictly to the requested window
    df = df[(df["timeSec"] >= start_ts) & (df["timeSec"] < end_ts)]

    return df


def fetch_trades_range(
    ex: ccxt.Exchange,
    symbol: str,
    start_ts: int,
    end_ts: int,
    limit: int = 1000,
    max_batches: int = 2000,
) -> pd.DataFrame:
    """Best-effort pagination for recent trades where supported. Many venues restrict lookback."""
    since = _ms(start_ts)
    rows: List[Dict] = []
    batches = 0
    while True:
        try:
            trades = ex.fetch_trades(symbol, since=since, limit=limit)
        except ccxt.BaseError:
            _sleep(ex); continue
        if not trades:
            break
        for t in trades:
            ts = t.get("timestamp")
            if ts and ts < _ms(end_ts):
                rows.append({
                    "timestamp": ts,
                    "price": t.get("price"),
                    "amount": t.get("amount"),
                    "cost": t.get("cost"),
                    "side": t.get("side"),
                    "id": t.get("id"),
                    "order": t.get("order"),
                    "takerOrMaker": t.get("takerOrMaker"),
                })
        last_ts = trades[-1].get("timestamp") if trades else None
        if last_ts is None or last_ts >= _ms(end_ts) - 1:
            break
        since = last_ts + 1
        batches += 1
        if batches >= max_batches:
            break
        _sleep(ex)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["exchange"] = ex.id
    df["symbol"] = symbol

    # Standardize time column to common schema (seconds since epoch)
    df["timeSec"] = (pd.to_numeric(df["timestamp"], errors="coerce") // 1000).astype("Int64")

    # Sort, de-dupe, and clip to window
    df = df.drop_duplicates().sort_values("timeSec").reset_index(drop=True)
    df = df[(df["timeSec"] >= start_ts) & (df["timeSec"] < end_ts)]

    return df


def snapshot_orderbook(
    ex: ccxt.Exchange,
    symbol: str,
    depth: int = 50,
) -> pd.DataFrame:
    """Fetch current orderbook (not historical)."""
    try:
        ob = ex.fetch_order_book(symbol, limit=depth)
    except ccxt.BaseError:
        return pd.DataFrame()

    ts_ms = ob.get("timestamp")
    if ts_ms is None:
        ts_ms = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)

    bids = pd.DataFrame(ob.get("bids", []), columns=["price", "amount"]); bids["side"] = "bid"
    asks = pd.DataFrame(ob.get("asks", []), columns=["price", "amount"]); asks["side"] = "ask"
    df = pd.concat([bids, asks], axis=0, ignore_index=True)

    df["timestamp"] = ts_ms
    df["timeSec"] = int(ts_ms // 1000)
    df["exchange"] = ex.id
    df["symbol"] = symbol

    # No sorting: keep exchange-provided order (price levels as returned)
    return df


# ----------------------------
# Orchestrator
# ----------------------------
@dataclass
class TaskSpec:
    exchange_id: str
    symbols: List[str]
    timeframe: str = DEFAULT_TIMEFRAME
    start_ts: int = START
    end_ts: int = END
    fetch_trades: bool = False
    snapshot_ob: bool = False


def run_cex_extract_ccxt(
    exchanges: Optional[List[str]] = None,
    symbols: Optional[List[str]] = None,
    timeframe: str = DEFAULT_TIMEFRAME,
    start_ts: int = START,
    end_ts: int = END,
    fetch_trades: bool = False,
    snapshot_ob: bool = False,
) -> None:
    """Collect raw CEX data (OHLCV + optional trades/orderbook snapshots) into data/raw/cex."""
    eids = exchanges if exchanges is not None else EXCHANGES
    syms = symbols if symbols is not None else SYMBOLS

    if start_ts == START and end_ts == END:
        s = dt.datetime.utcfromtimestamp(START).strftime("%Y-%m-%d")
        e = dt.datetime.utcfromtimestamp(END).strftime("%Y-%m-%d")
        print(f"Window: full ({s} → {e}, UTC)")

    # Save a simple meta snapshot (markets available per exchange)
    meta_rows: List[Dict] = []

    for eid in eids:
        print(f"\n=== {eid} ===")
        try:
            ex = _make_exchange(eid)
        except Exception as e:
            # Fallback for Huobi rebrand (some environments use 'htx')
            if eid == "huobi":
                try:
                    ex = _make_exchange("htx")
                    print("[INFO] fell back to 'htx' for Huobi.")
                except Exception as e2:
                    print(f"[WARN] cannot init {eid}/htx: {e2}")
                    continue
            else:
                print(f"[WARN] cannot init {eid}: {e}")
                continue

        # Apply simple per-exchange symbol aliases (e.g., Kraken XBT)
        syms_ex = _alias_symbols_for_exchange(ex.id, syms)

        # Keep only symbols listed on this exchange
        supported = [s for s in syms_ex if s in ex.markets]

        # Handle empty supported list
        if not supported:
            print("[INFO] no supported symbols.")
            continue

        # Meta
        for s in supported:
            m = ex.markets.get(s, {})
            meta_rows.append({
                "exchange": ex.id,
                "symbol": s,
                "base": m.get("base"),
                "quote": m.get("quote"),
                "active": m.get("active"),
                "spot": m.get("spot"),
                "type": m.get("type"),
                "timeframe": timeframe,
            })

        # OHLCV
        for s in supported:
            outp_dir = OHLCV_DIR / ex.id
            outp_dir.mkdir(parents=True, exist_ok=True)
            tag = f"{s.replace('/', '-')}_{timeframe}.parquet"
            outp = outp_dir / tag
            if outp.exists():
                print(f"[SKIP] OHLCV {ex.id}:{s} ({timeframe}) exists.")
            else:
                print(f"[OHLCV] {ex.id}:{s} {dt.datetime.utcfromtimestamp(start_ts).date()}→{dt.datetime.utcfromtimestamp(end_ts).date()}")
                df = fetch_ohlcv_range(ex, s, timeframe, start_ts, end_ts)
                if not df.empty:
                    df.to_parquet(outp, index=False, compression="snappy")
                    print(f"Saved -> {outp}  (rows={len(df)}, first={pd.to_datetime(df['timeSec'].min(), unit='s', utc=True).date()}, last={pd.to_datetime(df['timeSec'].max(), unit='s', utc=True).date()})")
                else:
                    print("[INFO] no OHLCV returned.")

        # Trades (best-effort)
        if fetch_trades:
            for s in supported:
                outt_dir = TRADES_DIR / ex.id
                outt_dir.mkdir(parents=True, exist_ok=True)
                tag = f"{s.replace('/', '-')}.parquet"
                outt = outt_dir / tag
                if outt.exists():
                    print(f"[SKIP] trades {ex.id}:{s} exist.")
                else:
                    print(f"[TRADES] {ex.id}:{s} best-effort {dt.datetime.utcfromtimestamp(start_ts).date()}→{dt.datetime.utcfromtimestamp(end_ts).date()}")
                    td = fetch_trades_range(ex, s, start_ts, end_ts)
                    if not td.empty:
                        td.to_parquet(outt, index=False, compression="snappy")
                        print(f"Saved -> {outt}  (rows={len(td)})")
                    else:
                        print("[INFO] trades unavailable/limited.")

        # Orderbook snapshot (now)
        if snapshot_ob:
            for s in supported:
                ob_dir = OB_DIR / ex.id
                ob_dir.mkdir(parents=True, exist_ok=True)
                ob = snapshot_orderbook(ex, s, depth=50)
                if not ob.empty:
                    ts_iso = dt.datetime.utcfromtimestamp(int(ob["timeSec"].iloc[0])).strftime("%Y%m%dT%H%M%SZ")
                    tag = f"{s.replace('/', '-')}_{ts_iso}.parquet"
                    outo = ob_dir / tag
                    ob.to_parquet(outo, index=False, compression="snappy")
                    print(f"[OB] saved snapshot -> {outo}")
                else:
                    print("[INFO] orderbook snapshot failed.")

    if meta_rows:
        meta_path = META_DIR / "markets.parquet"
        pd.DataFrame(meta_rows).drop_duplicates().to_parquet(meta_path, index=False, compression="snappy")
        print(f"\nSaved markets meta -> {meta_path}")

if __name__ == "__main__":
    run_cex_extract_ccxt()
