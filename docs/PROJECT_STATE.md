
# DEX Trading & Liquidity Intelligence — Project State

Updated: 2026-10-28

## Current Status

Phase 1: Initial on-chain data pipeline and trading analytics completed.

Study A: Historical execution quality analysis and single-range matched execution pilot completed.

Current development environment:
- Local project: /Users/dingminzhao/Desktop/dex
- VS Code / macOS / Apple M1 8GB
- Conda: dex-intel, Python 3.11
- GitHub: DINGMINZHAO/dex-trading-liquidity-intelligence
- Development architecture: Local-first
- Storage: Parquet; local DuckDB available for future analytical work
- Colab High-RAM: Optional, only when required

## Completed Work

### Data Engineering

- The Graph API configured through local .env
- Two Uniswap V3 USDC/WETH pools validated
- September 2026 daily pool dataset: 60 records
- September 1–3 swap pilot: 21,390 Swap events
- Cursor-based pagination and checkpoint recovery
- Compressed raw data and processed Parquet
- Duplicate-event and transaction-log validation
- Daily trading-volume reconciliation: PASS
- Zero-leg events preserved and flagged

### Trading Analytics

- Daily pool overview
- Trade size distribution
- Intraday activity and trading volume
- Buy/Sell direction and volume imbalance

### Execution Intelligence

- Observed execution cost by fee tier and size
- Fee-adjusted execution penalties
- Terminal pool-price movements
- Historical pool-state query feasibility
- Single-range matched swap quote simulation
- Multi-block execution comparison across 12 snapshots

## Research Dataset

Blockchain: Ethereum Mainnet
Protocol: Uniswap V3
Pools: USDC/WETH, fee tiers 0.05% and 0.30%
Pilot: 2026-09-01 to 2026-09-04 UTC (exclusive end)

Swap events: 21,390
Unique transaction hashes: 20,485
Valid observed-cost sample (>= $100): 13,757
Zero-leg events: 2

Historical simulation:
- 12 snapshots at six-hour intervals
- Order sizes: $1K, $5K, $10K, $20K
- Direction: USDC to WETH
- Single-range model
- Tick-boundary crossing excluded
- Gas, routing, and MEV excluded

## Current Findings

- 0.05% pool has substantially higher trading volume.
- 0.30% pool has a higher median Swap size.
- Larger observed swaps generally exhibit higher execution costs.
- Pool-price differences can partially offset fee advantages.
- In the matched pilot, the 0.05% pool wins most comparable cases.
- Simulation coverage decreases at larger order sizes.

## Figures

01 — Daily Pool Overview
02 — Trade Size Distribution
03 — Intraday Trading Activity
04 — Observed Execution Cost
05 — Price Impact Analysis
06 — Single-block Matched Quotes
07 — Multi-block Matched Execution

## Known Limitations

- Three-day initial research sample.
- Historical simulation currently supports single-range execution only.
- Uniswap V3 pool states may change across initialized ticks.
- No transaction-level router reconstruction or MEV attribution.
- No gas-inclusive execution-cost comparison yet.
- Same-subgraph volume reconciliation is not independent RPC verification.

## Next Priorities

1. Commit scripts, figures, README, state, and pilot report.
2. Expand historical execution analysis selectively.
3. Test robustness and improve tick-aware quoting where necessary.
4. Develop LP and liquidity analytics.
5. Build an interactive dashboard.
6. Prepare final portfolio documentation.

## Development Rules

- VS Code is the primary code editing environment.
- GitHub is the formal version-control source.
- Prefer local Python, Parquet, and DuckDB.
- Keep initial datasets small; avoid unnecessary data engineering.
- Never commit API secrets or large raw datasets.
- Report actual results separately from simulated results.
- Prioritize empirical findings, visualizations, and business relevance.
