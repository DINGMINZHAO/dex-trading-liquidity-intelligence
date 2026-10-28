
import argparse
import json
import math
import os
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
OUT = DATA_ROOT / "processed" / "swaps_2026q3"
MANIFEST = DATA_ROOT / "_system" / "swaps_2026q3_manifest.json"
PILOT = DATA_ROOT / "processed" / "swaps_pilot_20260901_20260904.parquet"
DAILY = DATA_ROOT / "processed" / "pool_daily_sep2026.parquet"

SUBGRAPH = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
URL = f"https://gateway.thegraph.com/api/subgraphs/id/{SUBGRAPH}"

POOLS = {
    "0.05%": ("0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640", 500),
    "0.30%": ("0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8", 3000),
}

USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
PAGE = 1000


def partition_path(day, fee):
    return OUT / f"pool_{fee:04d}" / f"{day}.parquet"


def save_partition(df, path, day, label):
    if df.empty:
        raise RuntimeError(f"Empty partition: {day} {label}")

    if df["event_id"].duplicated().any():
        raise RuntimeError(f"Duplicate event IDs: {day} {label}")

    valid_logs = df.dropna(subset=["log_index"])
    if valid_logs.duplicated(["transaction_hash", "log_index"]).any():
        raise RuntimeError(f"Duplicate tx/log index: {day} {label}")

    actual_days = pd.to_datetime(
        df["timestamp_utc"], utc=True
    ).dt.strftime("%Y-%m-%d")

    if not (actual_days == day).all() or not (df["fee_tier"] == label).all():
        raise RuntimeError(f"Wrong date or pool: {day} {label}")

    if not pd.to_numeric(
        df["trade_size_usd"], errors="coerce"
    ).ge(0).all():
        raise RuntimeError(f"Invalid USD volume: {day} {label}")

    df = df.copy()
    df["log_index"] = df["log_index"].astype("Int64")
    df["tick_after"] = df["tick_after"].astype("Int64")

    df = df.sort_values([
        "timestamp_utc", "block_number", "log_index", "event_id"
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.parquet")

    df.to_parquet(temp, index=False, compression="zstd")
    os.replace(temp, path)


def seed_pilot():
    if not PILOT.exists():
        raise FileNotFoundError(f"Pilot dataset not found: {PILOT}")

    pilot = pd.read_parquet(PILOT)

    if pilot["event_id"].duplicated().any():
        raise RuntimeError("Duplicate events in pilot")

    pilot["day_key"] = pd.to_datetime(
        pilot["timestamp_utc"], utc=True
    ).dt.strftime("%Y-%m-%d")

    expected = {
        (d, label)
        for d in ("2026-09-01", "2026-09-02", "2026-09-03")
        for label in POOLS
    }

    actual = set(zip(pilot["day_key"], pilot["fee_tier"]))

    if expected != actual:
        raise RuntimeError(f"Pilot coverage mismatch: {actual ^ expected}")

    for (day, label), group in pilot.groupby(["day_key", "fee_tier"]):
        path = partition_path(day, POOLS[label][1])

        if not path.exists():
            save_partition(
                group.drop(columns="day_key"), path, day, label
            )
            print(
                f"PILOT REUSED | {day} {label}: {len(group):,} events",
                flush=True,
            )


def build_session():
    key = os.getenv("GRAPH_API_KEY")

    if not key:
        raise RuntimeError("GRAPH_API_KEY missing from .env")

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {key}"

    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
    )

    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def normalize(s, label, address, start, end):
    t = int(s["timestamp"])

    if not start <= t < end or s["pool"]["id"].lower() != address:
        raise RuntimeError(f"Unexpected date or pool: {s['id']}")

    a0 = Decimal(s["amount0"])
    a1 = Decimal(s["amount1"])
    zero = a0 == 0 or a1 == 0

    if a0 > 0 and a1 < 0:
        direction = "BUY_WETH"
    elif a0 < 0 and a1 > 0:
        direction = "SELL_WETH"
    elif zero:
        direction = "UNDETERMINED_ZERO_LEG"
    else:
        raise RuntimeError(f"Unexpected amount signs: {s['id']}")

    price = None if zero else float(abs(a0 / a1))
    volume = float(s["amountUSD"])

    if (
        not math.isfinite(volume)
        or volume < 0
        or (
            price is not None
            and (not math.isfinite(price) or price <= 0)
        )
    ):
        raise RuntimeError(f"Invalid trade value: {s['id']}")

    timestamp = pd.to_datetime(t, unit="s", utc=True)

    return {
        "event_id": s["id"],
        "transaction_hash": s["transaction"]["id"],
        "log_index": (
            int(s["logIndex"]) if s.get("logIndex") is not None else None
        ),
        "block_number": int(s["transaction"]["blockNumber"]),
        "timestamp_utc": timestamp,
        "date": timestamp.floor("D"),
        "pool_address": address,
        "fee_tier": label,
        "token0": USDC,
        "token1": WETH,
        "amount0_usdc": float(a0),
        "amount1_weth": float(a1),
        "trade_direction": direction,
        "is_zero_leg": zero,
        "trade_size_usd": volume,
        "execution_price_usdc_per_weth": price,
        "tick_after": (
            int(s["tick"]) if s.get("tick") is not None else None
        ),
        "sqrt_price_x96_after": s["sqrtPriceX96"],
        "sender": s.get("sender"),
        "recipient": s.get("recipient"),
        "origin": s.get("origin"),
    }


def fetch_day(session, day, label, address, fee):
    start = int(pd.Timestamp(day, tz="UTC").timestamp())
    end = start + 86400
    cursor = ""
    rows = []

    while True:
        query = f"""
        {{
          _meta {{ block {{ number }} hasIndexingErrors }}

          pool(id: "{address}") {{
            id
            feeTier
            token0 {{ id decimals }}
            token1 {{ id decimals }}
          }}

          swaps(
            first: {PAGE}
            orderBy: id
            orderDirection: asc
            where: {{
              pool: "{address}"
              timestamp_gte: {start}
              timestamp_lt: {end}
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

        response = session.post(
            URL, json={"query": query}, timeout=90
        )
        response.raise_for_status()
        result = response.json()

        if result.get("errors"):
            raise RuntimeError(result["errors"])

        data = result["data"]

        if data["_meta"]["hasIndexingErrors"]:
            raise RuntimeError("Subgraph indexing error")

        pool = data["pool"]

        if (
            pool is None
            or pool["id"].lower() != address
            or int(pool["feeTier"]) != fee
            or pool["token0"]["id"].lower() != USDC
            or pool["token1"]["id"].lower() != WETH
            or int(pool["token0"]["decimals"]) != 6
            or int(pool["token1"]["decimals"]) != 18
        ):
            raise RuntimeError("Pool/token metadata mismatch")

        batch = data["swaps"]
        ids = [s["id"] for s in batch]

        if (
            ids != sorted(ids)
            or len(ids) != len(set(ids))
            or (ids and cursor and ids[0] <= cursor)
        ):
            raise RuntimeError(
                f"Invalid pagination order: {day} {label}"
            )

        rows.extend(
            normalize(s, label, address, start, end)
            for s in batch
        )

        if not batch or len(batch) < PAGE:
            break

        cursor = ids[-1]

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--september-only",
        action="store_true",
        help="Finish September before expanding July-August",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    seed_pilot()

    september = list(
        pd.date_range("2026-09-04", "2026-09-30", tz="UTC")
    )
    earlier = list(
        pd.date_range("2026-07-01", "2026-08-31", tz="UTC")
    )
    dates = september if args.september_only else september + earlier

    manifest = (
        json.loads(MANIFEST.read_text())
        if MANIFEST.exists() else {}
    )

    manifest.update({
        "window_start_utc": "2026-07-01",
        "window_end_exclusive_utc": "2026-10-01",
        "source": "The Graph Uniswap V3 subgraph",
        "subgraph_id": SUBGRAPH,
    })

    records = manifest.setdefault("partitions", {})

    for ts in pd.date_range("2026-09-01", "2026-09-03", tz="UTC"):
        day = ts.strftime("%Y-%m-%d")

        for label, (_, fee) in POOLS.items():
            key = f"{day}/{fee}"

            if key not in records:
                pilot_part = pd.read_parquet(
                    partition_path(day, fee),
                    columns=["event_id", "trade_size_usd"],
                )
                records[key] = {
                    "events": len(pilot_part),
                    "volume_usd": round(
                        float(pilot_part["trade_size_usd"].sum()), 2
                    ),
                    "reused_pilot": True,
                }

    daily_lookup = {}

    if DAILY.exists():
        daily = pd.read_parquet(
            DAILY, columns=["date", "fee_tier", "volume_usd"]
        )
        daily["day_key"] = pd.to_datetime(
            daily["date"], utc=True
        ).dt.strftime("%Y-%m-%d")

        daily_lookup = {
            (x.day_key, x.fee_tier): float(x.volume_usd)
            for x in daily.itertuples()
        }

    work = [
        (ts.strftime("%Y-%m-%d"), label, address, fee)
        for ts in dates
        for label, (address, fee) in POOLS.items()
    ]

    session = None

    try:
        for index, (day, label, address, fee) in enumerate(work, 1):
            path = partition_path(day, fee)
            key = f"{day}/{fee}"

            if path.exists():
                if key not in records:
                    cols = pd.read_parquet(
                        path,
                        columns=["event_id", "trade_size_usd"],
                    )
                    records[key] = {
                        "events": len(cols),
                        "volume_usd": round(
                            float(cols["trade_size_usd"].sum()), 2
                        ),
                    }

                print(
                    f"[{index}/{len(work)}] SKIP {day} {label} "
                    f"({records[key]['events']:,} events)",
                    flush=True,
                )
            else:
                if session is None:
                    session = build_session()

                df = fetch_day(
                    session, day, label, address, fee
                )
                save_partition(df, path, day, label)

                records[key] = {
                    "events": len(df),
                    "volume_usd": round(
                        float(df["trade_size_usd"].sum()), 2
                    ),
                }

                print(
                    f"[{index}/{len(work)}] SAVED {day} {label} "
                    f"({len(df):,} events)",
                    flush=True,
                )

            if (day, label) in daily_lookup:
                volume = daily_lookup[(day, label)]
                gap_pct = (
                    abs(records[key]["volume_usd"] / volume - 1) * 100
                    if volume > 0 else None
                )

                records[key]["daily_volume_gap_pct"] = (
                    round(gap_pct, 4)
                    if gap_pct is not None else None
                )

                if gap_pct is not None and gap_pct > 2:
                    print(
                        f"  WARNING: daily volume gap {gap_pct:.2f}%",
                        flush=True,
                    )

            manifest["last_updated_utc"] = (
                datetime.now(timezone.utc).isoformat()
            )

            manifest["completed_pool_days"] = sum(
                partition_path(
                    ts.strftime("%Y-%m-%d"), pool_fee
                ).exists()
                for ts in pd.date_range(
                    "2026-07-01", "2026-09-30", tz="UTC"
                )
                for _, pool_fee in POOLS.values()
            )

            temp = MANIFEST.with_suffix(".tmp")
            temp.write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )
            os.replace(temp, MANIFEST)

    finally:
        if session:
            session.close()

    all_days = pd.date_range(
        "2026-07-01", "2026-09-30", tz="UTC"
    )

    total = sum(
        partition_path(
            ts.strftime("%Y-%m-%d"), fee
        ).exists()
        for ts in all_days
        for _, fee in POOLS.values()
    )

    september_total = sum(
        partition_path(
            ts.strftime("%Y-%m-%d"), fee
        ).exists()
        for ts in pd.date_range(
            "2026-09-01", "2026-09-30", tz="UTC"
        )
        for _, fee in POOLS.values()
    )

    print(f"\nQ3 coverage: {total}/184 pool-days")
    print(f"September coverage: {september_total}/60 pool-days")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
