
import gzip
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_ROOT = Path(os.getenv("DEX_DATA_ROOT", ROOT / "data"))
SUBGRAPH = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"

POOLS = {
    "0.05%": (
        "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
        500,
    ),
    "0.30%": (
        "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",
        3000,
    ),
}

USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"

START_DATE = "2026-09-01"
END_DATE = "2026-09-04"  # Exclusive UTC

START = int(pd.Timestamp(START_DATE, tz="UTC").timestamp())
END = int(pd.Timestamp(END_DATE, tz="UTC").timestamp())

PAGE_SIZE = 1000
MAX_PER_POOL = 100_000

STEM = "swaps_pilot_20260901_20260904"

RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
SYSTEM_DIR = DATA_ROOT / "_system"

PARTIAL_PATH = RAW_DIR / f"{STEM}.jsonl.gz.partial"
RAW_PATH = RAW_DIR / f"{STEM}.jsonl.gz"
PARQUET_PATH = PROCESSED_DIR / f"{STEM}.parquet"
MANIFEST_PATH = SYSTEM_DIR / f"{STEM}_manifest.json"


def make_query(address, cursor):
    if cursor and not re.fullmatch(
        r"0x[0-9a-f]{64}#[0-9]+", cursor
    ):
        raise ValueError(f"Invalid cursor: {cursor}")

    return f"""
    {{
      _meta {{
        block {{ number }}
        hasIndexingErrors
      }}

      pool(id: "{address}") {{
        id
        feeTier
        token0 {{ id decimals }}
        token1 {{ id decimals }}
      }}

      swaps(
        first: {PAGE_SIZE}
        orderBy: id
        orderDirection: asc
        where: {{
          pool: "{address}"
          timestamp_gte: {START}
          timestamp_lt: {END}
          id_gt: "{cursor}"
        }}
      ) {{
        id
        timestamp
        transaction {{ id blockNumber }}
        pool {{ id }}
        sender
        recipient
        origin
        amount0
        amount1
        amountUSD
        sqrtPriceX96
        tick
        logIndex
      }}
    }}
    """


def normalize(swap, label):
    address = POOLS[label][0]

    if swap["pool"]["id"].lower() != address:
        raise ValueError(f"Wrong pool: {swap['id']}")

    timestamp = int(swap["timestamp"])

    if not START <= timestamp < END:
        raise ValueError(f"Invalid timestamp: {swap['id']}")

    a0 = Decimal(swap["amount0"])
    a1 = Decimal(swap["amount1"])

    zero_leg = a0 == 0 or a1 == 0

    if a0 > 0 and a1 < 0:
        direction = "BUY_WETH"
    elif a0 < 0 and a1 > 0:
        direction = "SELL_WETH"
    elif zero_leg:
        direction = "UNDETERMINED_ZERO_LEG"
    else:
        raise ValueError(
            f"Unexpected token signs: {swap['id']}"
        )

    if zero_leg:
        execution_price = None
    else:
        execution_price = float(abs(a0 / a1))

    trade_size = float(swap["amountUSD"])

    if not math.isfinite(trade_size) or trade_size < 0:
        raise ValueError(f"Invalid trade size: {swap['id']}")

    if execution_price is not None:
        if not math.isfinite(execution_price) or execution_price <= 0:
            raise ValueError(f"Invalid price: {swap['id']}")

    log_index = swap.get("logIndex")
    tick = swap.get("tick")

    return {
        "event_id": swap["id"],
        "transaction_hash": swap["transaction"]["id"],
        "log_index": int(log_index) if log_index is not None else None,
        "block_number": int(swap["transaction"]["blockNumber"]),
        "timestamp_utc": pd.to_datetime(
            timestamp, unit="s", utc=True
        ),
        "pool_address": address,
        "fee_tier": label,
        "token0": USDC,
        "token1": WETH,
        "amount0_usdc": float(a0),
        "amount1_weth": float(a1),
        "trade_direction": direction,
        "is_zero_leg": zero_leg,
        "trade_size_usd": trade_size,
        "execution_price_usdc_per_weth": execution_price,
        "tick_after": int(tick) if tick is not None else None,
        "sqrt_price_x96_after": swap["sqrtPriceX96"],
        "sender": swap["sender"],
        "recipient": swap["recipient"],
        "origin": swap["origin"],
    }


def load_checkpoint():
    rows = []
    seen = set()
    cursors = {label: "" for label in POOLS}
    counts = {label: 0 for label in POOLS}

    if not PARTIAL_PATH.exists():
        print("No checkpoint found. Starting from beginning.")
        return rows, seen, cursors, counts

    print("Existing checkpoint detected.")

    with gzip.open(
        PARTIAL_PATH, "rt", encoding="utf-8"
    ) as f:
        for line in f:
            record = json.loads(line)
            label = record.pop("fee_tier")

            if label not in POOLS:
                raise ValueError(f"Unknown fee tier: {label}")

            event_id = record["id"]

            if event_id in seen:
                raise RuntimeError(
                    f"Duplicate event in checkpoint: {event_id}"
                )

            if cursors[label] and event_id <= cursors[label]:
                raise RuntimeError(
                    f"Checkpoint not sorted by ID: {event_id}"
                )

            rows.append(normalize(record, label))
            seen.add(event_id)
            cursors[label] = event_id
            counts[label] += 1

    print(f"Recovered {len(rows):,} existing events")

    for label in POOLS:
        print(f"  {label}: {counts[label]:,}")

    return rows, seen, cursors, counts


def build_session():
    key = os.getenv("GRAPH_API_KEY")

    if not key:
        raise RuntimeError("GRAPH_API_KEY missing from .env")

    session = requests.Session()

    session.headers.update({
        "Authorization": f"Bearer {key}"
    })

    adapter = HTTPAdapter(
        max_retries=Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )
    )

    session.mount("https://", adapter)
    return session


def fetch_page(session, address, cursor, expected_fee):
    url = (
        "https://gateway.thegraph.com/api/"
        f"subgraphs/id/{SUBGRAPH}"
    )

    response = session.post(
        url,
        json={"query": make_query(address, cursor)},
        timeout=90,
    )

    response.raise_for_status()
    result = response.json()

    if result.get("errors"):
        raise RuntimeError(result["errors"])

    data = result["data"]

    if data["_meta"]["hasIndexingErrors"]:
        raise RuntimeError("Subgraph indexing errors")

    pool = data["pool"]

    if (
        pool is None
        or pool["id"].lower() != address
        or int(pool["feeTier"]) != expected_fee
        or pool["token0"]["id"].lower() != USDC
        or pool["token1"]["id"].lower() != WETH
        or int(pool["token0"]["decimals"]) != 6
        or int(pool["token1"]["decimals"]) != 18
    ):
        raise ValueError("Pool metadata mismatch")

    return data


def fetch_remaining(rows, seen, cursors, counts):
    session = build_session()
    indexed_blocks = {}
    new_count = 0

    try:
        with gzip.open(
            PARTIAL_PATH, "at", encoding="utf-8"
        ) as raw_file:

            for label, (address, fee) in POOLS.items():
                cursor = cursors[label]
                pages = 0

                print(
                    f"\nResuming {label}: "
                    f"{counts[label]:,} saved",
                    flush=True,
                )

                while True:
                    data = fetch_page(
                        session, address, cursor, fee
                    )

                    indexed_blocks[label] = int(
                        data["_meta"]["block"]["number"]
                    )

                    batch = data["swaps"]

                    if not batch:
                        print(f"{label}: complete")
                        break

                    ids = [s["id"] for s in batch]

                    if (
                        ids != sorted(ids)
                        or len(ids) != len(set(ids))
                        or (cursor and ids[0] <= cursor)
                    ):
                        raise RuntimeError(
                            f"Pagination ordering error: {label}"
                        )

                    if counts[label] + len(batch) > MAX_PER_POOL:
                        raise RuntimeError(
                            f"Safety limit exceeded: {label}"
                        )

                    for swap in batch:
                        event_id = swap["id"]

                        if event_id in seen:
                            raise RuntimeError(
                                f"Duplicate event: {event_id}"
                            )

                        row = normalize(swap, label)

                        raw_file.write(
                            json.dumps({
                                "fee_tier": label,
                                **swap,
                            }) + "\n"
                        )

                        rows.append(row)
                        seen.add(event_id)
                        cursor = event_id
                        counts[label] += 1
                        new_count += 1

                    raw_file.flush()
                    pages += 1

                    print(
                        f"{label} | new page {pages:02d} "
                        f"| total {counts[label]:,}",
                        flush=True,
                    )

                    if len(batch) < PAGE_SIZE:
                        print(f"{label}: complete")
                        break

        return indexed_blocks, new_count

    finally:
        session.close()


def validate_and_save(rows, counts, indexed_blocks, new_count):
    if not rows:
        raise RuntimeError("No swap events available")

    df = pd.DataFrame(rows)

    if df["event_id"].duplicated().any():
        raise RuntimeError("Duplicate event IDs")

    valid_logs = df.dropna(subset=["log_index"])

    if valid_logs.duplicated(
        ["transaction_hash", "log_index"]
    ).any():
        raise RuntimeError("Duplicate transaction + log index")

    if df["trade_size_usd"].isna().any():
        raise RuntimeError("Missing trade size")

    if (df["trade_size_usd"] < 0).any():
        raise RuntimeError("Negative trade size")

    df["log_index"] = df["log_index"].astype("Int64")
    df["tick_after"] = df["tick_after"].astype("Int64")

    df = df.sort_values([
        "timestamp_utc",
        "block_number",
        "log_index",
        "event_id",
    ]).reset_index(drop=True)

    df["date"] = df["timestamp_utc"].dt.floor("D")

    summary = df.groupby(
        ["date", "fee_tier"]
    ).agg(
        swap_events=("event_id", "size"),
        unique_transactions=("transaction_hash", "nunique"),
        zero_leg_events=("is_zero_leg", "sum"),
        swap_volume_usd=("trade_size_usd", "sum"),
    ).reset_index()

    print("\n=== Three-Day Swap Activity ===")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:,.2f}",
        )
    )

    if len(summary) != 6:
        raise RuntimeError(
            "Expected six pool-day combinations"
        )

    daily_path = (
        DATA_ROOT / "processed"
        / "pool_daily_sep2026.parquet"
    )

    reconciliation = "NOT_CHECKED"
    max_gap_pct = None

    if daily_path.exists():
        daily = pd.read_parquet(daily_path)[
            ["date", "fee_tier", "volume_usd"]
        ]

        merged = summary.merge(
            daily,
            on=["date", "fee_tier"],
            how="left",
            validate="one_to_one",
        )

        merged["volume_gap_pct"] = (
            100 * (
                merged["swap_volume_usd"]
                / merged["volume_usd"]
                - 1
            )
        )

        print("\n=== Daily Volume Reconciliation ===")
        print(
            merged[
                [
                    "date",
                    "fee_tier",
                    "swap_volume_usd",
                    "volume_usd",
                    "volume_gap_pct",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:,.4f}",
            )
        )

        if merged["volume_gap_pct"].notna().all():
            max_gap_pct = float(
                merged["volume_gap_pct"].abs().max()
            )

            reconciliation = (
                "PASS" if max_gap_pct <= 2 else "REVIEW"
            )
        else:
            reconciliation = "REVIEW"

    temp_parquet = PROCESSED_DIR / f"{STEM}.tmp.parquet"

    df.to_parquet(
        temp_parquet,
        index=False,
        compression="zstd",
    )

    os.replace(temp_parquet, PARQUET_PATH)

    try:
        git_commit = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None

    manifest = {
        "source": "Uniswap V3 Subgraph via The Graph",
        "subgraph_id": SUBGRAPH,
        "start_utc": START_DATE,
        "end_exclusive_utc": END_DATE,
        "fetched_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "git_head_at_run": git_commit,
        "row_count": len(df),
        "recovered_checkpoint_rows": len(df) - new_count,
        "newly_fetched_rows": new_count,
        "rows_by_pool": counts,
        "indexed_blocks_during_resume": indexed_blocks,
        "zero_leg_events": int(df["is_zero_leg"].sum()),
        "unique_transactions": int(
            df["transaction_hash"].nunique()
        ),
        "null_log_indexes": int(
            df["log_index"].isna().sum()
        ),
        "structural_checks": "PASS",
        "daily_volume_reconciliation": reconciliation,
        "max_daily_volume_gap_pct": max_gap_pct,
        "data_notes": (
            "Zero-leg swaps retained; direction undetermined "
            "and effective execution price set to null. "
            "Swap event count is not user trade count."
        ),
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    os.replace(PARTIAL_PATH, RAW_PATH)

    print("\n=== Final Results ===")
    print("Structural checks: PASS")
    print(f"Volume reconciliation: {reconciliation}")
    print(f"Total swap events: {len(df):,}")
    print(f"Newly fetched: {new_count:,}")
    print(f"Zero-leg events: {manifest['zero_leg_events']:,}")
    print(
        "Unique transactions: "
        f"{manifest['unique_transactions']:,}"
    )
    print(f"Parquet: {PARQUET_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")


def main():
    print(
        f"Pilot: {START_DATE} to {END_DATE} UTC "
        "(end exclusive)"
    )

    for directory in (
        RAW_DIR, PROCESSED_DIR, SYSTEM_DIR
    ):
        directory.mkdir(parents=True, exist_ok=True)

    if RAW_PATH.exists() and not PARTIAL_PATH.exists():
        raise RuntimeError(
            "Completed raw dataset already exists. "
            "Do not overwrite it without reviewing."
        )

    rows, seen, cursors, counts = load_checkpoint()

    indexed_blocks, new_count = fetch_remaining(
        rows, seen, cursors, counts
    )

    validate_and_save(
        rows, counts, indexed_blocks, new_count
    )


if __name__ == "__main__":
    main()
