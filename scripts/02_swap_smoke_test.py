
import gzip
import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_ROOT = Path(os.getenv("DEX_DATA_ROOT", ROOT / "data"))
SUBGRAPH = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"

POOLS = {
    "0.05%": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
    "0.30%": "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",
}

USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"

START = int(pd.Timestamp("2026-09-01", tz="UTC").timestamp())
END = int(pd.Timestamp("2026-09-04", tz="UTC").timestamp())
LIMIT = 20

QUERY = """
{
  _meta {
    block { number }
    hasIndexingErrors
  }

  pool(id: "$POOL") {
    id
    feeTier
    token0 { id decimals }
    token1 { id decimals }
  }

  swaps(
    first: $LIMIT
    orderBy: timestamp
    orderDirection: asc
    where: {
      pool: "$POOL"
      timestamp_gte: $START
      timestamp_lt: $END
    }
  ) {
    id
    timestamp
    transaction { id blockNumber }
    pool { id }
    sender
    recipient
    origin
    amount0
    amount1
    amountUSD
    sqrtPriceX96
    tick
    logIndex
  }
}
"""


def fetch_swaps(address):
    key = os.getenv("GRAPH_API_KEY")
    if not key:
        raise RuntimeError("GRAPH_API_KEY is missing from .env")

    query = (
        QUERY.replace("$POOL", address)
        .replace("$LIMIT", str(LIMIT))
        .replace("$START", str(START))
        .replace("$END", str(END))
    )

    url = (
        "https://gateway.thegraph.com/api/"
        f"subgraphs/id/{SUBGRAPH}"
    )

    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}"},
        json={"query": query},
        timeout=60,
    )
    response.raise_for_status()

    result = response.json()
    if result.get("errors"):
        raise RuntimeError(f"GraphQL error: {result['errors']}")

    data = result["data"]

    if data["_meta"]["hasIndexingErrors"]:
        raise RuntimeError("Subgraph reports indexing errors")

    pool = data["pool"]
    if pool is None or pool["id"].lower() != address:
        raise ValueError(f"Pool not found: {address}")

    if (
        pool["token0"]["id"].lower() != USDC
        or pool["token1"]["id"].lower() != WETH
        or int(pool["token0"]["decimals"]) != 6
        or int(pool["token1"]["decimals"]) != 18
    ):
        raise ValueError("Unexpected token ordering or decimals")

    return result


def normalize_swaps(result, fee_tier, address):
    rows = []

    for swap in result["data"]["swaps"]:
        amount0 = Decimal(swap["amount0"])
        amount1 = Decimal(swap["amount1"])

        if amount0 > 0 and amount1 < 0:
            direction = "BUY_WETH"
        elif amount0 < 0 and amount1 > 0:
            direction = "SELL_WETH"
        else:
            raise ValueError(f"Invalid swap signs: {swap['id']}")

        timestamp = int(swap["timestamp"])
        if not START <= timestamp < END:
            raise ValueError(f"Swap outside window: {swap['id']}")

        if swap["pool"]["id"].lower() != address:
            raise ValueError(f"Wrong pool: {swap['id']}")

        rows.append({
            "event_id": swap["id"],
            "transaction_hash": swap["transaction"]["id"],
            "log_index": (
                int(swap["logIndex"])
                if swap["logIndex"] is not None
                else None
            ),
            "block_number": int(swap["transaction"]["blockNumber"]),
            "timestamp_utc": pd.to_datetime(
                timestamp, unit="s", utc=True
            ),
            "pool_address": address,
            "fee_tier": fee_tier,
            "token0": USDC,
            "token1": WETH,
            "amount0_usdc": float(amount0),
            "amount1_weth": float(amount1),
            "trade_direction": direction,
            "trade_size_usd": float(swap["amountUSD"]),
            "execution_price_usdc_per_weth": float(
                abs(amount0 / amount1)
            ),
            "tick_after": int(swap["tick"]),
            "sqrt_price_x96_after": swap["sqrtPriceX96"],
            "sender": swap["sender"],
            "recipient": swap["recipient"],
            "origin": swap["origin"],
        })

    return rows


def main():
    raw_results = {}
    rows = []

    for fee_tier, address in POOLS.items():
        print(f"Fetching {fee_tier} pool...")

        result = fetch_swaps(address)
        raw_results[fee_tier] = result

        pool = result["data"]["pool"]
        expected_fee = 500 if fee_tier == "0.05%" else 3000
        if int(pool["feeTier"]) != expected_fee:
            raise ValueError(f"Unexpected fee tier: {fee_tier}")

        pool_rows = normalize_swaps(result, fee_tier, address)
        rows.extend(pool_rows)

        print(f"  Retrieved {len(pool_rows)} swap events")

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No swaps returned")

    if df["event_id"].duplicated().any():
        raise RuntimeError("Duplicate event IDs detected")

    if df["trade_size_usd"].isna().any():
        raise RuntimeError("Missing trade sizes")

    if (df["trade_size_usd"] < 0).any():
        raise RuntimeError("Negative trade sizes")

    df = df.sort_values(
        ["timestamp_utc", "block_number", "event_id"]
    ).reset_index(drop=True)

    for folder in ("raw", "processed", "_system"):
        (DATA_ROOT / folder).mkdir(parents=True, exist_ok=True)

    name = "swaps_smoke_20260901_20260904"

    raw_path = DATA_ROOT / "raw" / f"{name}.json.gz"
    parquet_path = DATA_ROOT / "processed" / f"{name}.parquet"
    manifest_path = DATA_ROOT / "_system" / f"{name}_manifest.json"

    with gzip.open(raw_path, "wt", encoding="utf-8") as f:
        json.dump(raw_results, f)

    df.to_parquet(
        parquet_path,
        index=False,
        compression="zstd",
    )

    manifest = {
        "source": "Uniswap V3 Subgraph via The Graph",
        "subgraph_id": SUBGRAPH,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_utc": "2026-09-01",
        "end_exclusive_utc": "2026-09-04",
        "sample_limit_per_pool": LIMIT,
        "sample_only": True,
        "row_count": len(df),
        "indexed_blocks": {
            label: result["data"]["_meta"]["block"]["number"]
            for label, result in raw_results.items()
        },
        "quality_checks": "PASS",
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\n=== Data Quality: PASS ===")
    print(f"Total rows: {len(df)}")

    print("\n=== Swap Samples ===")
    print(
        df[
            [
                "timestamp_utc",
                "fee_tier",
                "trade_direction",
                "trade_size_usd",
                "execution_price_usdc_per_weth",
            ]
        ].groupby("fee_tier").head(5).to_string(index=False)
    )

    print(f"\nParquet: {parquet_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
