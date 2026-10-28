
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

sys.path.insert(0, str(ROOT / "scripts"))
quote = importlib.import_module("09_matched_swap_quote")

DATA_ROOT = Path(os.getenv("DEX_DATA_ROOT", ROOT / "data"))
INPUT = DATA_ROOT / "processed/swaps_pilot_20260901_20260904.parquet"

CACHE = DATA_ROOT / "processed/historical_pool_states_6h_sep2026.json"
OUTPUT = DATA_ROOT / "analytics/matched_quotes_multiblock_pilot.csv"

START = pd.Timestamp("2026-09-01", tz="UTC")
END = pd.Timestamp("2026-09-04", tz="UTC")

SIZES = [1000, 5000, 10000, 20000]

SUBGRAPH = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"

QUERY = """
query HistoricalPools($block: Int!) {
  low: pool(
    id: "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
    block: {number: $block}
  ) {
    id feeTier liquidity sqrtPrice tick
    token0 { id decimals }
    token1 { id decimals }
  }

  high: pool(
    id: "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
    block: {number: $block}
  ) {
    id feeTier liquidity sqrtPrice tick
    token0 { id decimals }
    token1 { id decimals }
  }
}
"""


def select_snapshots():
    df = pd.read_parquet(INPUT)

    required = {
        "timestamp_utc", "block_number", "event_id"
    }

    if required - set(df.columns):
        raise ValueError("Missing required columns")

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], utc=True
    )

    if df["event_id"].duplicated().any():
        raise ValueError("Duplicate event IDs")

    df = df.sort_values(
        ["timestamp_utc", "block_number"]
    ).reset_index(drop=True)

    targets = pd.date_range(
        START, END, freq="6h", inclusive="left"
    )

    selected = []

    for target in targets:
        candidates = df[
            df["timestamp_utc"] >= target
        ]

        if candidates.empty:
            raise ValueError(f"No events after {target}")

        event = candidates.iloc[0]
        actual = event["timestamp_utc"]

        if actual - target > pd.Timedelta(minutes=15):
            raise ValueError(
                f"Snapshot too far from target: {target}"
            )

        selected.append({
            "target_utc": target.isoformat(),
            "snapshot_utc": actual.isoformat(),
            "block_number": int(event["block_number"]),
        })

    blocks = [s["block_number"] for s in selected]

    if len(blocks) != len(set(blocks)):
        raise ValueError("Duplicate snapshot blocks")

    return selected


def fetch_states(snapshots):
    expected_blocks = [
        s["block_number"] for s in snapshots
    ]

    if CACHE.exists():
        payload = json.loads(CACHE.read_text())

        saved_blocks = [
            s["block_number"] for s in payload
        ]

        if saved_blocks != expected_blocks:
            raise ValueError("Cached snapshot blocks mismatch")

        print("Using saved historical states")
        return payload

    key = os.getenv("GRAPH_API_KEY")

    if not key:
        raise RuntimeError("Missing GRAPH_API_KEY")

    url = (
        "https://gateway.thegraph.com/api/"
        f"subgraphs/id/{SUBGRAPH}"
    )

    results = []

    with requests.Session() as session:
        session.headers.update({
            "Authorization": f"Bearer {key}"
        })

        for i, snap in enumerate(snapshots, start=1):
            block = snap["block_number"]

            response = session.post(
                url,
                json={
                    "query": QUERY,
                    "variables": {"block": block},
                },
                timeout=90,
            )

            response.raise_for_status()
            payload = response.json()

            if payload.get("errors"):
                raise RuntimeError(
                    f"Block {block}: {payload['errors']}"
                )

            data = payload["data"]

            if not data.get("low") or not data.get("high"):
                raise RuntimeError(
                    f"Missing historical state: {block}"
                )

            results.append({
                **snap,
                "pools": data,
            })

            print(
                f"Historical snapshot {i}/12: "
                f"block {block}",
                flush=True,
            )

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    return results


def build_state(pool, label):
    config = quote.POOLS[label]

    if pool["id"].lower() != config["address"]:
        raise ValueError(f"Wrong pool address: {label}")

    if int(pool["feeTier"]) != config["fee"]:
        raise ValueError(f"Wrong fee tier: {label}")

    token0 = pool["token0"]
    token1 = pool["token1"]

    if (
        token0["id"].lower()
        != "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
        or token1["id"].lower()
        != "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
        or int(token0["decimals"]) != 6
        or int(token1["decimals"]) != 18
    ):
        raise ValueError("Unexpected token metadata")

    liquidity = int(pool["liquidity"])
    sqrt_x96 = int(pool["sqrtPrice"])
    tick = int(pool["tick"])

    if liquidity <= 0 or sqrt_x96 <= 0:
        raise ValueError("Invalid historical pool state")

    spacing = config["tick_spacing"]
    lower_tick = (tick // spacing) * spacing

    lower_sqrt = quote.tick_sqrt_x96(lower_tick)
    upper_sqrt = quote.tick_sqrt_x96(
        lower_tick + spacing
    )

    if not lower_sqrt <= sqrt_x96 <= upper_sqrt:
        raise ValueError("Inconsistent tick and price")

    return {
        "liquidity": liquidity,
        "sqrt_x96": sqrt_x96,
        "tick": tick,
        "lower_tick": lower_tick,
        "lower_sqrt": lower_sqrt,
        "spot_price": quote.price_usdc_per_weth(sqrt_x96),
    }


def run_study(snapshots):
    records = []

    for snapshot in snapshots:
        block = snapshot["block_number"]

        quote.BLOCK = block

        states = {
            "0.05%": build_state(
                snapshot["pools"]["low"], "0.05%"
            ),
            "0.30%": build_state(
                snapshot["pools"]["high"], "0.30%"
            ),
        }

        reference_price = (
            float(states["0.05%"]["spot_price"])
            + float(states["0.30%"]["spot_price"])
        ) / 2

        for amount in SIZES:
            low = quote.simulate_exact_input_usdc(
                "0.05%", states["0.05%"], amount
            )

            high = quote.simulate_exact_input_usdc(
                "0.30%", states["0.30%"], amount
            )

            comparable = (
                low["status"] == "OK_SINGLE_RANGE"
                and high["status"] == "OK_SINGLE_RANGE"
            )

            advantage_weth = np.nan
            advantage_usd = np.nan
            winner = "NOT_COMPARABLE"

            if comparable:
                advantage_weth = (
                    low["weth_received"]
                    - high["weth_received"]
                )

                advantage_usd = (
                    advantage_weth * reference_price
                )

                winner = (
                    "0.05%" if advantage_weth > 0
                    else "0.30%" if advantage_weth < 0
                    else "TIE"
                )

            records.append({
                "target_utc": snapshot["target_utc"],
                "snapshot_utc": snapshot["snapshot_utc"],
                "block_number": block,
                "input_usdc": amount,
                "low_status": low["status"],
                "high_status": high["status"],
                "low_weth": low["weth_received"],
                "high_weth": high["weth_received"],
                "low_effective_price": low["execution_price"],
                "high_effective_price": high["execution_price"],
                "low_advantage_weth": advantage_weth,
                "low_advantage_usd": advantage_usd,
                "winner": winner,
            })

    return pd.DataFrame(records)


def main():
    selected = select_snapshots()

    print(
        f"Selected {len(selected)} snapshots "
        "at 6-hour intervals"
    )

    historical = fetch_states(selected)
    results = run_study(historical)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)

    print("\n=== Multi-block Matched Quote Summary ===")

    for amount in SIZES:
        part = results[
            results["input_usdc"] == amount
        ]

        valid = part[
            part["winner"] != "NOT_COMPARABLE"
        ]

        low_wins = (valid["winner"] == "0.05%").sum()
        high_wins = (valid["winner"] == "0.30%").sum()
        ties = (valid["winner"] == "TIE").sum()

        print(f"\nOrder size: ${amount:,}")
        print(f"Comparable snapshots: {len(valid)}/12")
        print(f"0.05% wins: {low_wins}")
        print(f"0.30% wins: {high_wins}")
        print(f"Ties: {ties}")

        if not valid.empty:
            print(
                "Median 0.05% advantage: "
                f"${valid['low_advantage_usd'].median():+.4f}"
            )

    print(f"\nSaved: {OUTPUT}")
    print("Multi-block pilot complete.")


if __name__ == "__main__":
    main()
