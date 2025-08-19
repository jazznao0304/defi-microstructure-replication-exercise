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
