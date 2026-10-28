
# DEX Execution Intelligence — Pilot Findings

**Protocol:** Uniswap V3, Ethereum Mainnet  
**Markets:** USDC/WETH, 0.05% and 0.30% fee tiers  
**Period:** September 1–3, 2026 (UTC)

## Overview

This pilot examines trading behavior and execution quality across two Uniswap V3 liquidity pools.

The analysis uses 21,390 actual pool-level Swap events and historical liquidity pool states retrieved through The Graph.

The work covers trade-size distributions, trading direction, observed execution costs, price impact, and hypothetical matched-order execution.

## Key Findings

### 1. Market activity differs substantially across fee tiers

The 0.05% pool processed approximately $238.56 million in trading volume, compared with $17.90 million in the 0.30% pool.

Despite its lower total trading activity, the 0.30% pool had a larger median Swap size ($3,658 versus $1,434).

Trades between $10K and $100K accounted for approximately 64–66% of trading volume in both pools.

### 2. Execution costs increase with transaction size

Observed execution costs generally rose with Swap size.

After adjusting for the nominal fee, Buy-WETH trades of $10K–$100K had median execution penalties of approximately 1.11 bps in the 0.05% pool and 1.76 bps in the 0.30% pool.

The terminal pool-price movement was approximately twice the average execution-price penalty in many trade-size groups.

These results are consistent with the price impact of trading against an AMM liquidity curve.

### 3. Fee tiers alone do not determine the best execution venue

Historical pool states show that the two pools can have different starting prices.

Consequently, the fee advantage of the 0.05% pool may be partially offset by a less favorable initial exchange rate.

This motivates matched-order simulations rather than relying only on comparisons of observed trades.

### 4. Matched simulations generally favor the lower-fee pool

The pilot simulated identical USDC-to-WETH orders across 12 historical pool-state snapshots.

The 0.05% pool delivered a better simulated quote in:

- 11 of 12 comparable snapshots for $1K orders
- 12 of 12 for $5K orders
- 11 of 11 for $10K orders
- 8 of 8 for $20K orders

Median advantages were 6.26, 6.38, 9.59, and 12.44 bps of input value, respectively.

One $1K simulation favored the higher-fee pool, demonstrating that pool-price differences can reverse the expected ranking.

## Business Implications

The results support three practical considerations for DEX execution analytics.

First, quoted swap fees are only one component of transaction execution costs.

Second, optimal pool selection depends on current exchange rates and liquidity conditions.

Third, execution comparison tools should evaluate the same hypothetical order across available venues and clearly report when a quote cannot be reliably simulated.

These findings are relevant to DEX routing analytics, liquidity monitoring, and institutional digital asset execution research.

## Methodology and Scope

The underlying observations are actual Uniswap V3 subgraph data.

Observed execution-price penalties use reconstructed pre-Swap pool-price references.

Matched-order results come from a single-range historical simulation. The current simulator excludes cases that cross tick-spacing boundaries and does not model gas, router behavior, or MEV.

The study covers three days and provides an initial empirical benchmark. Further historical sampling and tick-aware simulation are planned before drawing broader conclusions.
