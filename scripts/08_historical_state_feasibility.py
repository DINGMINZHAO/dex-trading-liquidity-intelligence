
import json
import os
from decimal import Decimal, getcontext
from pathlib import Path

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

SUBGRAPH = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
BLOCK = 25889250

POOLS = {
    "low": {
        "address": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
        "fee": 500,
    },
    "high": {
        "address": "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",
        "fee": 3000,
    },
}

USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"

getcontext().prec = 60

QUERY = """
query HistoricalPools($block: Int!) {
  low: pool(
    id: "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
    block: {number: $block}
  ) {
    id
    feeTier
    liquidity
    sqrtPrice
    tick
    token0 { id decimals }
    token1 { id decimals }
  }

  high: pool(
    id: "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
    block: {number: $block}
  ) {
    id
    feeTier
    liquidity
    sqrtPrice
    tick
    token0 { id decimals }
    token1 { id decimals }
  }
}
"""


def spot_price(sqrt_price):
    ratio = Decimal(sqrt_price) / Decimal(2**96)
    return Decimal(10) ** 12 / (ratio * ratio)


def main():
    key = os.getenv("GRAPH_API_KEY")

    if not key:
        raise RuntimeError("Missing GRAPH_API_KEY")

    url = (
        "https://gateway.thegraph.com/api/"
        f"subgraphs/id/{SUBGRAPH}"
    )

    print(f"Testing historical pool state at block {BLOCK}")

    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "query": QUERY,
            "variables": {"block": BLOCK},
        },
        timeout=60,
    )

    response.raise_for_status()
    result = response.json()

    if result.get("errors"):
        print("\nHistorical query unavailable:")
        print(json.dumps(result["errors"], indent=2))
        return

    data = result.get("data") or {}

    for name, config in POOLS.items():
        pool = data.get(name)

        if pool is None:
            print(f"{name}: no historical pool state returned")
            return

        if (
            pool["id"].lower() != config["address"]
            or int(pool["feeTier"]) != config["fee"]
            or pool["token0"]["id"].lower() != USDC
            or pool["token1"]["id"].lower() != WETH
            or int(pool["token0"]["decimals"]) != 6
            or int(pool["token1"]["decimals"]) != 18
        ):
            raise ValueError(f"Pool metadata mismatch: {name}")

        if int(pool["liquidity"]) <= 0:
            raise ValueError(f"Invalid liquidity: {name}")

        if int(pool["sqrtPrice"]) <= 0:
            raise ValueError(f"Invalid price: {name}")

        print(f"\n=== {name} Pool ===")
        print("Fee Tier:", pool["feeTier"])
        print("Active Liquidity:", pool["liquidity"])
        print("Tick:", pool["tick"])
        print("sqrtPriceX96:", pool["sqrtPrice"])
        print(
            "Spot Price (USDC/WETH):",
            f"{spot_price(pool['sqrtPrice']):.6f}",
        )

    output = (
        ROOT / "data/processed"
        / f"historical_pool_state_block_{BLOCK}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({
            "block_number": BLOCK,
            "source": "Uniswap V3 Subgraph / The Graph",
            "pools": data,
        }, indent=2),
        encoding="utf-8",
    )

    print("\nHistorical state feasibility: PASS")
    print("Saved:", output)


if __name__ == "__main__":
    main()
