# ssrn-3984897-replication

Replication of SSRN 3984897.

## Folder layout
`

├── data
│   ├── raw          <- Vendor downloads and raw extracts (not tracked)
│   ├── interim      <- Cleaned / intermediate artifacts (not tracked)
│   └── processed    <- Final analysis tables (not tracked)
├── notebooks        <- Jupyter / Colab notebooks
├── src
│   └── replication  <- Python source code (functions, pipelines)
├── reports
│   └── figures      <- Generated plots for the paper
└── config           <- Credentials templates & settings

`

## Environment
Conda env: $CondaEnv (Python 3.11). See environment.yml.

<!-- REPL_EXPORT_START -->
# ssrn-3984897-replication

Replication of SSRN 3984897.

## Folder layout

```

├── data
│   ├── raw                  <- Vendor downloads and raw extracts (not tracked)
│   │   ├── cex/ohlcv
│   │   ├── ethereum/gas\_fees
│   │   ├── uniswap\_v2/pair\_day\_data
│   │   └── uniswap\_v3/pool\_day\_data
│   ├── interim              <- Cleaned / intermediate artifacts (not tracked)
│   │   ├── cex/proxies      <- Daily CEX microstructure proxies (Roll/CHL/CS/Amihud)
│   │   └── dex/proxies      <- Daily DEX proxies (Amihud/Roll) by v3 fee-tier & v2
│   └── processed            <- Final analysis tables (not tracked)
│       ├── compare          <- Daily comparison panels
│       │   ├── daily\_panel.parquet
│       │   └── daily\_panel\_agg.parquet   <- adds aggregated v3 ("uniswap\_v3\_all")
│       └── features         <- Modeling features
│           └── daily\_features.parquet
├── notebooks
│   ├── 01\_\* … 06\_\*          <- Data collection & panel build (up to daily panel)
│   ├── 07\_build\_features.ipynb
│   ├── 08\_descriptives.ipynb
│   ├── 09\_models\_fe\_did.ipynb
│   ├── 10\_robustness\_checks.ipynb
│   ├── 11\_visualization\_and\_interpretation.ipynb
│   └── 12\_export\_replication\_package.ipynb
├── src
│   └── replication
│       ├── build\_daily\_comparison\_panel.py
│       ├── cex\_proxies\_from\_ohlcv.py
│       └── dex\_proxies\_from\_daily.py
├── reports
│   ├── figures
│   │   ├── descriptives
│   │   ├── models
│   │   ├── robustness
│   │   └── interpretation
│   ├── models               <- Tidy coefficient tables (*.parquet)
│   ├── tables
│   │   ├── models           <- statsmodels summaries (*.txt)
│   │   └── robustness
│   ├── manifests            <- File inventories (auto-created)
│   └── system               <- Environment snapshots (pip/conda)
├── dist                     <- Zipped replication bundles
└── config                   <- Credentials templates & settings

```

## Environment

Conda env: `$CondaEnv` (Python 3.11). See `environment.yml`.
We also snapshot the environment during export into `reports/system/`.

---

## Research plan (what we replicate and why)

**Focus:** market microstructure under limited (daily) data. We proxy liquidity on CEX and DEX and study how Uniswap v3 and major shocks shift activity/liquidity.

**Events:**
* Uniswap v3 launch — 2021-05-05
* FTX collapse — 2022-11-10

**Main analyses (A1–A8):**
* **A1 — DEX activity vs v3 adoption (TWFE)**  
  *H1:* Higher v3 penetration within a pair increases DEX volume.  
  `log(DEX volume) ~ v3_share + ETH gas + FE(label) + FE(date)`
* **A2 — Spillovers to CEX volatility (TWFE)**  
  *H2:* Higher v3 share (DEX) lowers CEX volatility (abs returns).  
  `|CEX ret| ~ v3_share + ETH gas + FE(label) + FE(date)`
* **A3 — CEX microstructure (TWFE, per proxy)**  
  *H3:* CEX spreads/illiquidity (Roll, CHL, CS, Amihud) improve with volume and deteriorate with congestion.  
  `proxy_cex ~ log(CEX volume) + ETH gas + FEs`
* **A4 — DEX microstructure (TWFE, per proxy)**  
  *H4:* DEX Amihud/Roll decrease with DEX volume; increase with gas.  
  `proxy_dex ~ log(DEX volume) + ETH gas + FEs`
* **A5 — DiD #1: v3 launch (DEX vs CEX)**  
  *H5:* Post-launch, DEX volume rises relative to CEX.  
  `log(volume) ~ is_dex×Post_v3 + gas + FEs`
* **A6 — Event study: v3 (DEX only)**  
  *H6:* Flat pre-trends; post-event increase in DEX activity.
* **A7 — DiD #2: FTX (DEX vs CEX)**  
  *H7:* Post-FTX, DEX activity rises relative to CEX.
* **A8 — Event study: FTX (DEX only)**  
  *H8:* Persistent reallocation toward DEX post-FTX.

**Proxies used**
* **CEX (from OHLCV):** Roll (1984), CHL (Abdi–Ranaldo 2017), Corwin–Schultz (2012), Amihud (2002).
* **DEX (from daily OHLC + volume):** Amihud, Roll (adapted to daily).

---

## Core datasets (minimal to run models)

* `data/processed/compare/daily_panel.parquet` — CEX + DEX (v2 + v3 fee-tiers) + ETH gas.
* `data/processed/compare/daily_panel_agg.parquet` — adds aggregated v3 tier “uniswap_v3_all”.
* `data/processed/features/daily_features.parquet` — modeling features (e.g., `v3_share`, flags, logs, proxies).

---

## How to reproduce (quick)

1. Create the conda env (`environment.yml`) or use the snapshots in `reports/system/`.
2. Run or verify up to the processed panel (`build_daily_comparison_panel.py` / `06_*`).
3. Run `07_build_features.ipynb`.
4. Run `08_descriptives.ipynb` (optional figures).
5. Run `09_models_fe_did.ipynb` (FE + DiD) and `10_robustness_checks.ipynb`.
6. Inspect tables in `reports/tables/` and coefs in `reports/models/`.
7. (Optional) `11_visualization_and_interpretation.ipynb` for consolidated takeaways.
8. `12_export_replication_package.ipynb` to build a clean ZIP + README insert.

---

## Notes & limitations

* We work at **daily** frequency; true microstructure moments (trade/quote) are approximated by **proxies**. Some paper results cannot be exactly reproduced without high-freq data.
* Uniswap v3 fee-tier activity is aggregated to a **volume-weighted v3 share** per pair/day.
* Warnings like “covariance of constraints does not have full rank” arise from many FE dummies; clustered SEs are still computed but some dummy covariances are singular (expected under two-way FE with many fixed effects).

---

## Replication package

Run `12_export_replication_package.ipynb` to:
* snapshot the environment (pip/conda) into `reports/system/`,
* write a file manifest with hashes to `reports/manifests/`,
* zip essential artifacts into `dist/replication_package_*.zip`,
* update this section in place.

Artifacts included: processed panels, features, model coefs, model summaries, robustness tables, figures, and manifests (raw vendor downloads are excluded).
<!-- REPL_EXPORT_END -->




