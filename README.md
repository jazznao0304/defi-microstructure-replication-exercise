# ssrn-3984897-replication

Replication of SSRN 3984897 (scaled-down daily panel).
Focus: market microstructure around Uniswap v3 launch (2021-05-05) and FTX collapse (2022-11-10).

## Folder layout

```
├── config                 <- Credentials templates & settings (e.g., .env.template)
│
├── data
│   ├── raw                <- Vendor downloads and raw extracts (not tracked)
│   │   ├── uniswap_v2/pair_day_data
│   │   ├── uniswap_v3/pool_day_data
│   │   └── ethereum/gas_fees
│   ├── interim            <- Cleaned / intermediate artifacts (not tracked)
│   │   ├── cex/proxies    # CHL/CS/Amihud/Roll from CEX OHLCV (daily)
│   │   └── dex/proxies    # Amihud/Roll from DEX daily aggregates
│   └── processed          <- Final analysis tables (not tracked)
│       ├── compare        # daily_panel.parquet (+ v3-aggregated)
│       └── features       # daily_features.parquet
│
├── notebooks              <- Jupyter notebooks (repro order below)
│   ├── 01_... to 06_build_daily_comparison_panel.ipynb
│   ├── 07_build_features.ipynb
│   ├── 08_descriptives.ipynb
│   ├── 09_models_fe_did.ipynb
│   ├── 10_robustness_checks.ipynb
│   └── 11_visualization_and_interpretation.ipynb
│
├── reports
│   ├── figures
│   │   ├── descriptives
│   │   ├── robustness
│   │   └── interpretation
│   └── tables
│       ├── descriptives
│       └── models         # statsmodels summaries (.txt), tidy coefs (.parquet)
│
├── src
│   └── replication        <- Python source (pipelines + helpers)
│       ├── build_daily_comparison_panel.py
│       ├── cex_proxies_from_ohlcv.py
│       ├── dex_proxies_from_daily.py
│       └── (helpers, loaders, etc.)
│
└── environment.yml        <- Conda env (Python 3.11)
```

## Environment

Conda env: `mscqf-rep` (Python 3.11). See `environment.yml`.

```bash
conda env create -f environment.yml
conda activate mscqf-rep
python -V
```

## Reproduction outline (minimal)

1. **Data prep**

   * Place raw exports under `data/raw/...` (Uniswap v2/v3 daily, ETH gas fees).
   * (Optional) Build proxies

     * CEX: `src/replication/cex_proxies_from_ohlcv.py` → `data/interim/cex/proxies/...`
     * DEX: `src/replication/dex_proxies_from_daily.py` → `data/interim/dex/proxies/...`
2. **Panel build**

   * Run `src/replication/build_daily_comparison_panel.py`
     → `data/processed/compare/daily_panel.parquet` (+ `_agg` with v3 fee-tier aggregation)
3. **Features**

   * `07_build_features.ipynb` → `data/processed/features/daily_features.parquet`
4. **Descriptives**

   * `08_descriptives.ipynb` → plots/tables in `reports/figures/descriptives`, `reports/tables/descriptives`
5. **Models (FE & DiD)**

   * `09_models_fe_did.ipynb` → tidy coefs in `reports/models/*.parquet`, summaries in `reports/tables/models/*.txt`
6. **Robustness**

   * `10_robustness_checks.ipynb`
7. **Interpretation**

   * `11_visualization_and_interpretation.ipynb` → figures/tables under `reports/*`

## Research plan (brief)

* **Aim:** Replicate core findings with **daily** data and **microstructure proxies** (CEX: CHL/CS/Amihud/Roll; DEX: Amihud/Roll).
* **Main questions:**

  1. Does higher **Uniswap v3 penetration** raise **DEX activity**? (Panel TWFE)
  2. Do v3 gains **spill over** to **CEX volatility/liquidity**? (Panel TWFE with proxies)
  3. How do **events** (v3 launch, FTX collapse) shift activity **DEX vs CEX**? (DiD + event studies)

**Key events:** Uniswap v3 launch = 2021-05-05; FTX collapse = 2022-11-10.

## Key takeaways (very short)

* **v3 share → DEX volume (A1):** Positive and significant in TWFE (supports adoption → activity).
* **Spillovers to CEX vol (A2):** Inconclusive at daily frequency with our proxies.
* **CEX microstructure (A3):** Amihud behaves as expected (activity ↓ illiquidity). Roll/CHL/CS show mixed/opp signs—likely scale/measurement differences in daily proxies across venues.
* **DEX microstructure (A4):** Weak/inconclusive relationships with activity at daily frequency.
* **DiD (A5–A8):**

  * v3 launch DiD (DEX vs CEX) not robustly positive in our daily setup; DEX-only event study shows strong **pre-trends**, so treat causality cautiously.
  * FTX DiD: no clean reallocation toward DEX; DEX-only event study shows broad **post drop** (system-wide contraction).
* **Bottom line:** Daily + proxies replicate **directional** results for A1; event-driven causal claims are fragile without higher frequency and richer controls.

## Data policy

* `data/raw`, `data/interim`, and `data/processed` are **not tracked** in git (see `.gitignore`).
* Generated `reports/figures` & `reports/tables` **may** be tracked (optional).

## Quick start (notebooks)

Open in VS Code or Jupyter and run in order:

```
06_build_daily_comparison_panel.ipynb
07_build_features.ipynb
08_descriptives.ipynb
09_models_fe_did.ipynb
10_robustness_checks.ipynb
11_visualization_and_interpretation.ipynb
```

## Notes

* Event-study terms use `D_tau_mXX` / `D_tau_pXX` naming; plots show pre & post bins relative to $k=-1$.
* Standard errors are clustered by **label** (pair).
* All models include **pair FEs** and **date FEs** unless stated otherwise.
