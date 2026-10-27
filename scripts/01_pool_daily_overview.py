
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_ROOT = Path(os.getenv("DEX_DATA_ROOT", ROOT / "data"))

SUBGRAPH = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"

POOLS = {
    "low": (
        "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
        500,
        "0.05%",
    ),
    "high": (
        "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",
        3000,
        "0.30%",
    ),
}

START = "2026-09-01"
END = "2026-10-01"  # Exclusive, UTC

USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"


# ============================================================
# GraphQL Query
# ============================================================

QUERY = """
query Daily($start: Int!, $end: Int!) {
  _meta {
    block { number }
    hasIndexingErrors
  }

  lowPool: pool(
    id: "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
  ) {
    id
    feeTier
    token0 { id decimals }
    token1 { id decimals }
  }

  highPool: pool(
    id: "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
  ) {
    id
    feeTier
    token0 { id decimals }
    token1 { id decimals }
  }

  low: poolDayDatas(
    first: 100
    orderBy: date
    orderDirection: asc
    where: {
      pool: "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
      date_gte: $start
      date_lt: $end
    }
  ) {
    date
    volumeUSD
    tvlUSD
    txCount
  }

  high: poolDayDatas(
    first: 100
    orderBy: date
    orderDirection: asc
    where: {
      pool: "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
      date_gte: $start
      date_lt: $end
    }
  ) {
    date
    volumeUSD
    tvlUSD
    txCount
  }
}
"""


# ============================================================
# Data Ingestion
# ============================================================

def fetch_pool_data():
    key = os.getenv("GRAPH_API_KEY")

    if not key:
        raise RuntimeError(
            "GRAPH_API_KEY is missing. Check your .env file."
        )

    start = int(pd.Timestamp(START, tz="UTC").timestamp())
    end = int(pd.Timestamp(END, tz="UTC").timestamp())

    url = (
        "https://gateway.thegraph.com/api/"
        f"subgraphs/id/{SUBGRAPH}"
    )

    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "query": QUERY,
            "variables": {
                "start": start,
                "end": end,
            },
        },
        timeout=60,
    )

    response.raise_for_status()
    result = response.json()

    if result.get("errors"):
        raise RuntimeError(f"GraphQL error: {result['errors']}")

    data = result["data"]

    if data["_meta"]["hasIndexingErrors"]:
        raise RuntimeError("Subgraph reports indexing errors.")

    return result


# ============================================================
# Processing and Validation
# ============================================================

def process_pool_data(result):
    data = result["data"]
    rows = []

    expected_dates = pd.date_range(
        START,
        periods=30,
        freq="D",
        tz="UTC",
    )

    for name, (address, fee, label) in POOLS.items():

        pool = data[f"{name}Pool"]

        if pool is None:
            raise ValueError(f"Pool not found: {address}")

        assert pool["id"].lower() == address
        assert int(pool["feeTier"]) == fee

        assert pool["token0"]["id"].lower() == USDC
        assert pool["token1"]["id"].lower() == WETH

        assert int(pool["token0"]["decimals"]) == 6
        assert int(pool["token1"]["decimals"]) == 18

        for item in data[name]:
            rows.append({
                "date": pd.to_datetime(
                    int(item["date"]),
                    unit="s",
                    utc=True,
                ),
                "pool_address": address,
                "fee_tier": label,
                "volume_usd": float(item["volumeUSD"]),
                "tvl_usd": float(item["tvlUSD"]),
                "tx_count_subgraph": int(item["txCount"]),
            })

    df = pd.DataFrame(rows)

    df = df.sort_values(
        ["fee_tier", "date"]
    ).reset_index(drop=True)

    for label in ("0.05%", "0.30%"):

        part = df[df["fee_tier"] == label]

        assert len(part) == 30, (
            f"{label}: expected 30 days, got {len(part)}"
        )

        assert part["date"].is_unique

        assert part["date"].reset_index(drop=True).equals(
            pd.Series(expected_dates, name="date")
        ), f"{label}: missing or unexpected dates"

    numeric_columns = [
        "volume_usd",
        "tvl_usd",
        "tx_count_subgraph",
    ]

    assert np.isfinite(df[numeric_columns].to_numpy()).all()

    assert (df[numeric_columns] >= 0).all().all()

    assert df["volume_usd"].sum() > 0

    return df


# ============================================================
# Visualization
# ============================================================

def create_figure(df, output_path):

    colors = {
        "0.05%": "#167E86",
        "0.30%": "#D99143",
    }

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1, 1],
            "hspace": 0.28,
        },
    )

    series_config = [
        ("volume_usd", "Daily Trading Volume"),
        ("tvl_usd", "Pool Total Value Locked"),
    ]

    for ax, (column, title) in zip(axes, series_config):

        for label, color in colors.items():

            part = df[df["fee_tier"] == label]

            ax.plot(
                part["date"].dt.tz_localize(None),
                part[column] / 1e6,
                label=label,
                color=color,
                linewidth=2.3,
            )

        ax.set_title(
            title,
            loc="left",
            fontsize=13,
            fontweight="bold",
            pad=14,
        )

        ax.set_ylabel("USD millions", labelpad=12)

        ax.grid(
            axis="y",
            color="#E2E8F0",
            linewidth=0.8,
        )

        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_color("#CBD5E1")
        ax.spines["bottom"].set_color("#CBD5E1")

        ax.tick_params(
            axis="both",
            colors="#334155",
            length=0,
            pad=8,
        )

    # Fixed date ticks: no overlapping labels
    date_ticks = pd.date_range(
        START,
        periods=6,
        freq="5D",
    )

    axes[-1].set_xticks(date_ticks)
    axes[-1].xaxis.set_major_formatter(
        mdates.DateFormatter("%b %d")
    )

    axes[-1].set_xlim(
        pd.Timestamp(START),
        pd.Timestamp(END) - pd.Timedelta(days=1),
    )

    axes[-1].set_xlabel(
        "September 2026 (UTC)",
        labelpad=12,
    )

    axes[0].legend(
        loc="upper right",
        frameon=False,
        ncol=2,
        fontsize=10,
    )

    fig.suptitle(
        "Uniswap V3 | USDC/WETH Fee Tier Comparison",
        fontsize=19,
        fontweight="bold",
        x=0.075,
        y=0.97,
        ha="left",
        color="#13253A",
    )

    fig.text(
        0.075,
        0.025,
        (
            "Source: Uniswap V3 Subgraph / The Graph  |  "
            "September 1–30, 2026 (UTC)  |  "
            "TVL represents daily pool snapshots."
        ),
        fontsize=9,
        color="#64748B",
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.97,
        top=0.88,
        bottom=0.13,
    )

    fig.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():

    print("Fetching Uniswap V3 pool data...")

    result = fetch_pool_data()

    df = process_pool_data(result)

    for folder in (
        "raw",
        "processed",
        "exports",
        "_system",
    ):
        (DATA_ROOT / folder).mkdir(
            parents=True,
            exist_ok=True,
        )

    # Save original GraphQL response
    raw_path = DATA_ROOT / "raw/pool_daily_sep2026.json"

    raw_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    # Save processed data
    parquet_path = (
        DATA_ROOT / "processed/pool_daily_sep2026.parquet"
    )

    df.to_parquet(
        parquet_path,
        index=False,
        compression="zstd",
    )

    # Generate figure
    figure_path = (
        DATA_ROOT / "exports/01_pool_daily_overview.png"
    )

    create_figure(df, figure_path)

    # Record code version and data provenance
    commit = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()

    manifest = {
        "source": "Uniswap V3 Subgraph via The Graph",
        "subgraph_id": SUBGRAPH,
        "git_commit": commit,
        "indexed_block": result["data"]["_meta"]["block"]["number"],
        "fetched_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "start_utc": START,
        "end_exclusive_utc": END,
        "row_count": len(df),
        "quality_checks": "PASS",
        "pool_addresses": {
            name: info[0]
            for name, info in POOLS.items()
        },
    }

    manifest_path = (
        DATA_ROOT
        / "_system/pool_daily_sep2026_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    # Summary
    print("\nData quality: PASS")
    print(f"Rows: {len(df)}")
    print(f"Indexed block: {manifest['indexed_block']}")

    print("\nTrading volume by fee tier:")
    print(
        df.groupby("fee_tier")["volume_usd"]
        .sum()
        .div(1e6)
        .round(2)
        .to_string()
    )
    print("(USD millions, September 2026)")

    print(f"\nGit commit: {commit}")
    print(f"Parquet: {parquet_path}")
    print(f"Figure: {figure_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
