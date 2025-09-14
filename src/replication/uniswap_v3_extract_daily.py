import os
import time
import random
import datetime as dt
from typing import Dict, List, Tuple, Optional, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
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

UNISWAP_V3_SUBGRAPH = (
    f"https://gateway.thegraph.com/api/{THEGRAPH_API_KEY}/subgraphs/id/"
    "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
)

DATA_DIR      = ROOT / "data"
RAW_DIR       = DATA_DIR / "raw" / "uniswap_v3"
META_DIR      = RAW_DIR / "meta"
POOL_DAY_DIR  = RAW_DIR / "pool_day_data"

for d in (META_DIR, POOL_DAY_DIR):
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
    "KNC":   "0xdeFA4e8a7bcBA345F687a2f1456F5Edd9CE97202",  # New KNC (post-migration)
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
    """GraphQL client for the Uniswap v3 subgraph."""
    transport = RequestsHTTPTransport(
        url=UNISWAP_V3_SUBGRAPH,
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
# Fetch pools & poolDayData
# ----------------------------
def fetch_pools_for_pair(client: Client, tokenA: str, tokenB: str) -> List[dict]:
    """Return v3 pools for a token pair (both orderings) with feeTier and id."""
    q = gql("""
    query($a: Bytes!, $b: Bytes!) {
      p0: pools(first: 10, where: { token0: $a, token1: $b }) {
        id feeTier token0 { symbol decimals } token1 { symbol decimals }
      }
      p1: pools(first: 10, where: { token0: $b, token1: $a }) {
        id feeTier token0 { symbol decimals } token1 { symbol decimals }
      }
    }
    """)
    res = _safe_execute(client, q, {"a": tokenA.lower(), "b": tokenB.lower()})
    pools: List[dict] = []
    for k in ("p0", "p1"):
        if res.get(k):
            pools.extend(res[k])
    return pools

def fetch_pool_day_data(client: Client, pool_id: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """
    Fetch v3 poolDayDatas (daily prices/liquidity/volumes).
    CHUNK is a paging window (not granularity): we iterate in ~30-day slices to stay under API limits.
    Note: many deployments do NOT expose sqrtPrice on daily. We compute derived prices only if inputs exist.
    """
    rows: List[dict] = []
    q = gql("""
    query($pool: ID!, $start: Int!, $end: Int!, $skip: Int!) {
      poolDayDatas(first: 1000, skip: $skip,
        where: { pool: $pool, date_gte: $start, date_lt: $end },
        orderBy: date, orderDirection: asc) {
        date
        liquidity
        token0Price
        token1Price
        volumeToken0
        volumeToken1
        volumeUSD
        feesUSD
        pool { id token0 { symbol decimals } token1 { symbol decimals } feeTier }
      }
    }
    """)
    CHUNK = 30 * 24 * 3600  # Page in 30-day windows to keep responses <1k edges
    t0 = start_ts
    while t0 < end_ts:
        t1 = min(t0 + CHUNK, end_ts)
        skip = 0
        while True:
            data = _safe_execute(client, q, {"pool": pool_id, "start": t0, "end": t1, "skip": skip})
            part = data.get("poolDayDatas", [])
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

    # Flatten pool meta
    if "pool" in df.columns:
        meta = pd.json_normalize(df["pool"])
        meta.columns = [c.replace(".", "_") for c in meta.columns]
        df = pd.concat([df.drop(columns=["pool"]), meta], axis=1)

    # Force identity column (overwrite any meta-provided pool_id to be safe)
    df["pool_id"] = pool_id

    # Numeric casting (subgraph fields are BigDecimal strings in token units)
    num_cols = [
        "date", "liquidity",
        "token0Price", "token1Price",
        "volumeToken0", "volumeToken1", "volumeUSD", "feesUSD",
        "token0_decimals", "token1_decimals", "feeTier",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Reported prices (from subgraph)
    df["price_1_per_0_reported"] = df.get("token0Price")  # token1 per token0
    df["price_0_per_1_reported"] = df.get("token1Price")

    # Derived prices (token1 per token0) computed from sqrtPrice (Q64.96) and token decimals, if present
    if "sqrtPrice" in df.columns:
        sp = pd.to_numeric(df["sqrtPrice"], errors="coerce").astype("float64")
        ratio = (sp / (2.0 ** 96)) ** 2  # Raw ratio without decimals
        scale0 = 10.0 ** df["token0_decimals"].astype("float64")
        scale1 = 10.0 ** df["token1_decimals"].astype("float64")
        price_1_per_0 = ratio * (scale0 / scale1)  # token1 per token0
        df["price_1_per_0_derived"] = price_1_per_0.replace([float("inf")], pd.NA)
        df["price_0_per_1_derived"] = (1.0 / price_1_per_0).replace([float("inf")], pd.NA)
    else:
        df["price_1_per_0_derived"] = pd.NA
        df["price_0_per_1_derived"] = pd.NA

    # Lightweight QA (print-only; not saved)
    # --- Reciprocity: token0Price vs 1/token1Price
    r10 = pd.to_numeric(df.get("price_1_per_0_reported"), errors="coerce")  # token1 per token0
    r01 = pd.to_numeric(df.get("price_0_per_1_reported"), errors="coerce")  # token0 per token1
    m_recip = r10.notna() & r01.notna() & np.isfinite(r10) & np.isfinite(r01) & (r01 != 0)
    if m_recip.any():
        inv = 1.0 / r01[m_recip].astype("float64")
        recip_abs = (r10[m_recip].astype("float64") - inv).abs()
        if recip_abs.size > 0:
            med = float(np.nanmedian(recip_abs.to_numpy()))
            p95 = float(np.nanpercentile(recip_abs.to_numpy(), 95))
            print(f"   [delta] reciprocity median(abs): {med:.3e}, 95p: {p95:.3e}")

    # --- Reported vs Derived
    der = pd.to_numeric(df.get("price_1_per_0_derived"), errors="coerce")
    m_cmp = r10.notna() & der.notna() & np.isfinite(r10) & np.isfinite(der) & (r10 != 0)
    if m_cmp.any():
        dabs = (r10[m_cmp].astype("float64") - der[m_cmp].astype("float64")).abs()
        dpct = dabs / r10[m_cmp].astype("float64").abs()
        if dpct.size > 0:
            med = float(np.nanmedian(dpct.to_numpy()))
            p95 = float(np.nanpercentile(dpct.to_numpy(), 95))
            print(f"   [delta] reported vs derived median(pct): {med:.3e}, 95p: {p95:.3e}")

    # Standardize time column to common schema
    df.rename(columns={"date": "dayStartUnix"}, inplace=True)

    # Sort by dayStartUnix
    if "dayStartUnix" in df.columns:
        df = df.sort_values("dayStartUnix").reset_index(drop=True)

    return df


# ----------------------------
# Orchestrator
# ----------------------------
@dataclass
class PoolSpec:
    tokenA_symbol: str
    tokenB_symbol: str
    label: str
    pools: List[dict] = field(default_factory=list)

def run_uniswap_v3_extract_daily(
    start_ts: int = START,
    end_ts: int = END,
    labels: Optional[List[str]] = None,
    filename_suffix: str = "",
    out_subdir: str = "",
    fee_tiers: Optional[Iterable[int]] = (100, 500, 3000, 10000),
) -> None:
    """Fetch Uniswap v3 poolDayData; save Parquet under data/raw/uniswap_v3."""
    client = _gql_client()

    # Discover pools for each label
    specs: List[PoolSpec] = [PoolSpec(a, b, lbl) for (a, b, lbl) in PAIRS]
    for s in specs:
        a, b = TOKENS[s.tokenA_symbol], TOKENS[s.tokenB_symbol]
        s.pools = fetch_pools_for_pair(client, a, b)
        if s.pools:
            tiers = sorted({int(p.get("feeTier", -1)) for p in s.pools})
            print(f"[OK] {s.label} -> {len(s.pools)} v3 pool(s), tiers={tiers}")
    # Warn once for missing pools
    for s in specs:
        if not s.pools:
            print(f"[WARN] No v3 pools found: {s.label} ({s.tokenA_symbol}/{s.tokenB_symbol})")

    if start_ts == START and end_ts == END:
        s = dt.datetime.utcfromtimestamp(START).strftime("%Y-%m-%d")
        e = dt.datetime.utcfromtimestamp(END).strftime("%Y-%m-%d")
        print(f"Window: full ({s} → {e}, UTC)")

    # Save pool meta
    meta_rows = []
    for s in specs:
        for p in s.pools:
            meta_rows.append({
                "label": s.label,
                "tokenA": s.tokenA_symbol,
                "tokenB": s.tokenB_symbol,
                "tokenA_addr": TOKENS[s.tokenA_symbol],
                "tokenB_addr": TOKENS[s.tokenB_symbol],
                "pool_id": p["id"],
                "feeTier": int(p.get("feeTier", -1)),
                "token0_symbol": p["token0"]["symbol"],
                "token1_symbol": p["token1"]["symbol"],
                "token0_decimals": int(p["token0"]["decimals"]),
                "token1_decimals": int(p["token1"]["decimals"]),
            })
    meta_path = META_DIR / "pools.parquet"
    pd.DataFrame(meta_rows).to_parquet(meta_path, index=False)
    print(f"Saved pool map -> {meta_path}")

    day_dir = POOL_DAY_DIR / out_subdir if out_subdir else POOL_DAY_DIR
    day_dir.mkdir(parents=True, exist_ok=True)

    for s in specs:
        print(f"\n=== {s.label} ===")

        if labels is not None and s.label not in labels:
            print("Label filtered; skipping.")
            print(f"[DONE] {s.label}")
            continue

        if not s.pools:
            print(f"[WARN] No v3 pools found: {s.label} ({s.tokenA_symbol}/{s.tokenB_symbol})")
            print(f"[DONE] {s.label}")
            continue

        cand = [p for p in s.pools if int(p.get("feeTier", -1)) in set(fee_tiers)] if fee_tiers else list(s.pools)
        if not cand:
            print(f"[WARN] No pools in requested fee tiers for {s.label}; skipping.")
            print(f"[DONE] {s.label}")
            continue

        for p in cand:
            pid = p["id"]
            tier = int(p.get("feeTier", -1))
            tag  = f"{s.label.replace('/', '-').replace(' ', '_')}_fee{tier}{filename_suffix}.parquet"
            outp = day_dir / tag

            if outp.exists():
                print(f"Already have data for {s.label} (fee {tier}); skipping.")
            else:
                pdaily = fetch_pool_day_data(client, pid, start_ts, end_ts)
                if not pdaily.empty:
                    pdaily["pool_id"] = pid
                    pdaily["feeTier"] = tier
                    pdaily.to_parquet(outp, index=False)
                    print(f"Saved poolDayData -> {outp}")
                else:
                    print("No poolDayData returned for this window.")

        print(f"[DONE] {s.label}")

if __name__ == "__main__":
    run_uniswap_v3_extract_daily()
