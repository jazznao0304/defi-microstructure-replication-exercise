import os
import time
import random
import datetime as dt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport
from gql.transport.exceptions import TransportError, TransportQueryError
import requests


# ----------------------------
# Config
# ----------------------------
ROOT = Path(__file__).resolve().parents[2]  # .../ssrn-3984897-replication
ENV_PATH = ROOT / "config" / ".env"
load_dotenv(dotenv_path=ENV_PATH)

THEGRAPH_API_KEY = os.getenv("THEGRAPH_API_KEY")
if not THEGRAPH_API_KEY:
    raise RuntimeError(f"Missing THEGRAPH_API_KEY. Expected at: {ENV_PATH}")

UNISWAP_V2_SUBGRAPH = (
    f"https://gateway.thegraph.com/api/{THEGRAPH_API_KEY}/subgraphs/id/"
    "A3Np3RQbaBA6oKJgiwDJeo5T3zrYfGHPWFYayMwtNDum"
)

DATA_DIR      = ROOT / "data"
RAW_DIR       = DATA_DIR / "raw" / "uniswap_v2"
META_DIR      = RAW_DIR / "meta"
PAIR_HOUR_DIR = RAW_DIR / "pair_hour_data"
SWAPS_DIR     = RAW_DIR / "swaps"

for d in (META_DIR, PAIR_HOUR_DIR, SWAPS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Tokens & pairs
# ----------------------------
TOKENS: Dict[str, str] = {
    "WETH":  "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "USDC":  "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "USDT":  "0xdAC17F958D2ee523a2206206994597C13D831ec7",
    "WBTC":  "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
    "DAI":   "0x6B175474E89094C44Da98b954EedeAC495271d0F",
    "LINK":  "0x514910771AF9Ca656af840dff83E8264EcF986CA",
    "MANA":  "0x0F5D2fB29fb7d3CFeE444a200298f468908cC942",
    "AAVE":  "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9",
    "BAT":   "0x0D8775F648430679A709E98d2b0Cb6250d2887EF",
    "CRV":   "0xD533a949740bb3306d119CC777fa900bA034cd52",
    "GRT":   "0xc944E90C64B2c07662A292be6244BDf05Cda44a7",
    "KNC":   "0xdd974D5C2e2928deA5F71b9825b8b646686BD200",  # Legacy KNC (pre-migration)
    "REP":   "0x221657776846890989a759BA2973e427DfF5C9Bb",  # REPv2
    "SNX":   "0xC011a73ee8576Fb46F5E1c5751cA3B9Fe0aF2a6F",
    "STORJ": "0xB64ef51C888972c908CFacf59B47C1AfBC0Ab8aC",
    "UNI":   "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
    "OMG":   "0xd26114cd6EE289AccF82350c8d8487fedB8A0C07",
}

PAIRS: List[Tuple[str, str, str]] = [
    # 8-venue intersection
    ("WETH", "USDC", "ETH-USDC"),
    ("WETH", "USDT", "ETH-USDT"),
    ("WETH", "WBTC", "ETH-BTC"),
    ("LINK", "WETH", "LINK-ETH"),
    ("WBTC", "USDC", "BTC-USDC"),
    ("DAI",  "WETH", "DAI-ETH"),
    ("MANA", "WETH", "MANA-ETH"),
    ("USDC", "USDT", "USDC-USDT"),
    # Additional 12 for 4-exchange intersection
    ("DAI",  "USDT", "DAI-USDT"),
    ("AAVE", "WETH", "AAVE-ETH"),
    ("BAT",  "WETH", "BAT-ETH"),
    ("WBTC", "DAI",  "BTC-DAI"),
    ("CRV",  "WETH", "CRV-ETH"),
    ("GRT",  "WETH", "GRT-ETH"),
    ("KNC",  "WETH", "KNC-ETH"),
    ("REP",  "WETH", "REP-ETH"),
    ("SNX",  "WETH", "SNX-ETH"),
    ("STORJ","WETH", "STORJ-ETH"),
    ("UNI",  "WETH", "UNI-ETH"),
    ("OMG",  "WETH", "OMG-ETH"),
]

START = int(dt.datetime(2021, 2, 1, tzinfo=dt.timezone.utc).timestamp())  # Buffer for warm-up
END   = int(dt.datetime(2023, 3, 1, tzinfo=dt.timezone.utc).timestamp())


# ----------------------------
# Graph helpers
# ----------------------------
def _gql_client() -> Client:
    """GraphQL client for the Uniswap v2 subgraph."""
    transport = RequestsHTTPTransport(
        url=UNISWAP_V2_SUBGRAPH,
        timeout=60,
        verify=True,
        headers={"Content-Type": "application/json"},
    )
    return Client(transport=transport, fetch_schema_from_transport=False)

def _safe_execute(client: Client, query, variables, max_retries: int = 6, base_sleep: float = 0.75):
    """Execute a GraphQL query with simple exponential backoff."""
    for i in range(max_retries):
        try:
            return client.execute(query, variable_values=variables)
        except (TransportError, TransportQueryError, requests.RequestException):
            time.sleep(base_sleep * (2 ** i) + random.uniform(0, 0.25))
    return client.execute(query, variable_values=variables)


# ----------------------------
# Fetch pair, pairHourData & swaps
# ----------------------------
def fetch_pair_id(client: Client, tokenA: str, tokenB: str) -> Optional[str]:
    """Return the v2 pair ID for a token address pair (checks both orderings)."""
    q = gql("""
    query($a: Bytes!, $b: Bytes!) {
      pairs0: pairs(first: 1, where: {token0: $a, token1: $b}) { id }
      pairs1: pairs(first: 1, where: {token0: $b, token1: $a}) { id }
    }
    """)
    res = _safe_execute(client, q, {"a": tokenA.lower(), "b": tokenB.lower()})
    for k in ("pairs0", "pairs1"):
        if res.get(k) and len(res[k]) > 0:
            return res[k][0]["id"]
    return None

def fetch_pair_hour_data(client: Client, pair_id: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """
    Fetch v2 pairHourDatas (hourly aggregates).
    CHUNK is a paging window (not granularity): we iterate in ~7-day slices to stay under API limits.
    """
    rows: List[dict] = []
    q = gql("""
    query($pair: ID!, $start: Int!, $end: Int!, $skip: Int!) {
      pairHourDatas(first: 1000, skip: $skip,
        where: { pair: $pair, hourStartUnix_gte: $start, hourStartUnix_lt: $end },
        orderBy: hourStartUnix, orderDirection: asc) {
        hourStartUnix
        reserve0
        reserve1
        reserveUSD
        hourlyVolumeToken0
        hourlyVolumeToken1
        hourlyVolumeUSD
        pair { id token0 { symbol decimals } token1 { symbol decimals } }
      }
    }
    """)
    CHUNK = 7 * 24 * 3600  # Page in 7-day windows to keep responses <1k edges

    t0 = start_ts
    while t0 < end_ts:
        t1 = min(t0 + CHUNK, end_ts)
        skip = 0
        while True:
            data = _safe_execute(client, q, {"pair": pair_id, "start": t0, "end": t1, "skip": skip})
            part = data.get("pairHourDatas", [])
            if not part:
                break
            rows.extend(part)
            if len(part) < 1000:
                break
            skip += 1000
            time.sleep(0.1)
        t0 = t1

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["pair_id"] = pair_id

    # Flatten pair meta
    if "pair" in df.columns:
        meta = pd.json_normalize(df["pair"])
        meta.columns = [c.replace(".", "_") for c in meta.columns]
        df = pd.concat([df.drop(columns=["pair"]), meta], axis=1)

    # Force identity column (overwrite any meta-provided pair_id to be safe)
    df["pair_id"] = pair_id

    # Numeric casting (subgraph fields are BigDecimal strings in token units)
    for c in ["hourStartUnix", "reserve0", "reserve1", "reserveUSD",
              "hourlyVolumeToken0", "hourlyVolumeToken1", "hourlyVolumeUSD"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derived prices (token1 per token0) computed directly from reserves in token units
    if {"reserve0", "reserve1"}.issubset(df.columns):
        r0 = df["reserve0"].astype("float64")
        r1 = df["reserve1"].astype("float64")
        valid = r0.notna() & r1.notna() & (r0 != 0.0)
        px01 = (r1 / r0).where(valid).replace([float("inf"), -float("inf")], pd.NA)
        df["price_1_per_0_derived"] = px01
        df["price_0_per_1_derived"] = (1.0 / px01).replace([float("inf"), -float("inf")], pd.NA)
    else:
        df["price_1_per_0_derived"] = pd.NA
        df["price_0_per_1_derived"] = pd.NA

    # Reported prices not available on v2 entities
    df["price_1_per_0_reported"] = pd.NA
    df["price_0_per_1_reported"] = pd.NA

    # Standardize volume columns to common schema
    df["volumeToken0"] = pd.to_numeric(df.get("hourlyVolumeToken0"), errors="coerce")
    df["volumeToken1"] = pd.to_numeric(df.get("hourlyVolumeToken1"), errors="coerce")
    df["volumeUSD"]    = pd.to_numeric(df.get("hourlyVolumeUSD"), errors="coerce")

    # Sort by hourStartUnix
    if "hourStartUnix" in df.columns:
        df = df.sort_values("hourStartUnix").reset_index(drop=True)

    return df

def fetch_swaps(client: Client, pair_id: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """
    Fetch v2 swaps for a pair (timestamps and amounts).
    CHUNK is a paging window (not granularity): we iterate in ~3-day slices to stay under API limits.
    """
    rows: List[dict] = []
    q = gql("""
    query($pair: Bytes!, $start: Int!, $end: Int!, $skip: Int!) {
      swaps(first: 1000, skip: $skip,
            where: { pair: $pair, timestamp_gte: $start, timestamp_lt: $end },
            orderBy: timestamp, orderDirection: asc) {
        id
        amount0In
        amount1In
        amount0Out
        amount1Out
        amountUSD
        timestamp
        transaction { id }
      }
    }
    """)
    CHUNK = 3 * 24 * 3600  # Page in 3-day windows to keep responses <1k edges
    t0 = start_ts
    while t0 < end_ts:
        t1 = min(t0 + CHUNK, end_ts)
        skip = 0
        while True:
            data = _safe_execute(client, q, {"pair": pair_id, "start": t0, "end": t1, "skip": skip})
            part = data.get("swaps", [])
            if not part:
                break
            rows.extend(part)
            if len(part) < 1000:
                break
            skip += 1000
            time.sleep(0.1)
        t0 = t1

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["pair_id"] = pair_id
    if "transaction" in df.columns:
        tx = pd.json_normalize(df["transaction"])
        tx.columns = [f"tx_{c}" for c in tx.columns]
        df = pd.concat([df.drop(columns=["transaction"]), tx], axis=1)
    for c in ("timestamp", "amount0In", "amount1In", "amount0Out", "amount1Out", "amountUSD"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"], keep="last")
    return df


# ----------------------------
# Orchestrator
# ----------------------------
@dataclass
class PairSpec:
    tokenA_symbol: str
    tokenB_symbol: str
    label: str
    pair_id: Optional[str] = None

def run_uniswap_v2_extract_graph(
    start_ts: int = START,
    end_ts: int = END,
    labels: Optional[List[str]] = None,
    filename_suffix: str = "",
    out_subdir: str = "",
) -> None:
    """Fetch Uniswap v2 pairHourData and swaps; save Parquet under data/raw/uniswap_v2."""
    client = _gql_client()

    pairs: List[PairSpec] = [PairSpec(a, b, lbl) for (a, b, lbl) in PAIRS]
    for p in pairs:
        a, b = TOKENS[p.tokenA_symbol], TOKENS[p.tokenB_symbol]
        p.pair_id = fetch_pair_id(client, a, b)
        if p.pair_id:
            print(f"[OK] {p.label} -> {p.pair_id}")
    # Warn once for missing pair IDs
    for p in pairs:
        if not p.pair_id:
            print(f"[WARN] Pair not found on v2: {p.label} ({p.tokenA_symbol}/{p.tokenB_symbol})")

    if start_ts == START and end_ts == END:
        s = dt.datetime.utcfromtimestamp(START).strftime("%Y-%m-%d")
        e = dt.datetime.utcfromtimestamp(END).strftime("%Y-%m-%d")
        print(f"Window: full ({s} → {e}, UTC)")

    pair_map = pd.DataFrame(
        [{
            "label": p.label,
            "tokenA": p.tokenA_symbol,
            "tokenB": p.tokenB_symbol,
            "tokenA_addr": TOKENS[p.tokenA_symbol],
            "tokenB_addr": TOKENS[p.tokenB_symbol],
            "pair_id": p.pair_id,
        } for p in pairs]
    )
    meta_path = META_DIR / "pairs.parquet"
    pair_map.to_parquet(meta_path, index=False)
    print(f"Saved pair map -> {meta_path}")

    pair_dir  = PAIR_HOUR_DIR / out_subdir if out_subdir else PAIR_HOUR_DIR
    swaps_dir = SWAPS_DIR / out_subdir if out_subdir else SWAPS_DIR
    pair_dir.mkdir(parents=True, exist_ok=True)
    swaps_dir.mkdir(parents=True, exist_ok=True)

    for p in pairs:
        if not p.pair_id:
            continue
        if labels is not None and p.label not in labels:
            continue

        print(f"\n=== {p.label} ({p.pair_id}) ===")
        fname = f"{p.label.replace('/', '-').replace(' ', '_')}{filename_suffix}.parquet"
        outp = pair_dir / fname
        outs = swaps_dir / fname

        if outp.exists() and outs.exists():
            print(f"Already have data for {p.label}; skipping.")
        else:
            ph = fetch_pair_hour_data(client, p.pair_id, start_ts, end_ts)
            if not ph.empty:
                ph.to_parquet(outp, index=False)
                print(f"Saved pairHourData -> {outp}")
            else:
                print("No pairHourData returned for this window.")

            sw = fetch_swaps(client, p.pair_id, start_ts, end_ts)
            if not sw.empty:
                sw.to_parquet(outs, index=False)
                print(f"Saved swaps -> {outs}")
            else:
                print("No swaps returned for this window.")

        print(f"[DONE] {p.label}")

if __name__ == "__main__":
    run_uniswap_v2_extract_graph()
