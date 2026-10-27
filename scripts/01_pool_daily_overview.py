
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.getenv("DEX_DATA_ROOT", ROOT / "data"))
SUBGRAPH = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"

POOLS = {
    "low": ("0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640", 500, "0.05%"),
    "high": ("0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8", 3000, "0.30%"),
}

START = "2026-09-01"
END = "2026-10-01"  # Exclusive

QUERY = """
query Daily($start: Int!, $end: Int!) {
  _meta { block { number } hasIndexingErrors }

  lowPool: pool(id: "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640") {
    id feeTier
    token0 { id decimals }
    token1 { id decimals }
  }
  highPool: pool(id: "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8") {
    id feeTier
    token0 { id decimals }
    token1 { id decimals }
  }

  low: poolDayDatas(
    first: 100, orderBy: date, orderDirection: asc,
    where: {
      pool: "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
      date_gte: $start, date_lt: $end
    }
  ) { date volumeUSD tvlUSD txCount }

  high: poolDayDatas(
    first: 100, orderBy: date, orderDirection: asc,
    where: {
      pool: "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
      date_gte: $start, date_lt: $end
    }
  ) { date volumeUSD tvlUSD txCount }
}
"""

def main():
    key = os.environ["GRAPH_API_KEY"]
    start = int(pd.Timestamp(START, tz="UTC").timestamp())
    end = int(pd.Timestamp(END, tz="UTC").timestamp())

    url = f"https://gateway.thegraph.com/api/subgraphs/id/{SUBGRAPH}"
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}"},
        json={"query": QUERY, "variables": {"start": start, "end": end}},
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()

    if result.get("errors"):
        raise RuntimeError(result["errors"])

    data = result["data"]
    if data["_meta"]["hasIndexingErrors"]:
        raise RuntimeError("Subgraph reports indexing errors")

    rows = []
    usdc = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
    weth = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"

    for name, (address, fee, label) in POOLS.items():
        pool = data[f"{name}Pool"]
        assert pool is not None
        assert pool["id"].lower() == address
        assert int(pool["feeTier"]) == fee
        assert pool["token0"]["id"].lower() == usdc
        assert pool["token1"]["id"].lower() == weth
        assert int(pool["token0"]["decimals"]) == 6
        assert int(pool["token1"]["decimals"]) == 18

        for item in data[name]:
            rows.append({
                "date": pd.to_datetime(item["date"], unit="s", utc=True),
                "pool_address": address,
                "fee_tier": label,
                "volume_usd": float(item["volumeUSD"]),
                "tvl_usd": float(item["tvlUSD"]),
                "tx_count_subgraph": int(item["txCount"]),
            })

    df = pd.DataFrame(rows).sort_values(["fee_tier", "date"])

    for label in ("0.05%", "0.30%"):
        part = df[df["fee_tier"] == label]
        assert len(part) == 30, f"{label}: expected 30 days, got {len(part)}"
        assert part["date"].is_unique

    assert (df[["volume_usd", "tvl_usd", "tx_count_subgraph"]] >= 0).all().all()
    assert df["volume_usd"].sum() > 0

    for folder in ("raw", "processed", "exports", "_system"):
        (DATA_ROOT / folder).mkdir(parents=True, exist_ok=True)

    (DATA_ROOT / "raw/pool_daily_sep2026.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    df.to_parquet(DATA_ROOT / "processed/pool_daily_sep2026.parquet", index=False)

    colors = {"0.05%": "#167E86", "0.30%": "#D99143"}
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)

    fig.patch.set_facecolor("#FFFFFF")
    for ax, (column, title) in zip(
        axes,
        [("volume_usd", "Daily Trading Volume"),
         ("tvl_usd", "Pool Total Value Locked")],
    ):
        for label, color in colors.items():
            part = df[df["fee_tier"] == label]
            ax.plot(
                part["date"].dt.tz_localize(None),
                part[column] / 1e6,
                label=label,
                color=color,
                linewidth=2.2,
            )

        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=12)
        ax.set_ylabel("USD millions")
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#CBD5E1")

    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("September 2026 (UTC)")
    fig.suptitle(
        "Uniswap V3 | USDC/WETH Fee Tier Comparison",
        x=0.08, y=0.98, ha="left", fontsize=17, fontweight="bold",
    )
    fig.text(
        0.08, 0.015,
        "Source: Uniswap V3 Subgraph / The Graph | "
        "Daily indexed aggregates; TVL is a daily state snapshot.",
        fontsize=9, color="#64748B",
    )
    fig.tight_layout(rect=[0.06, 0.06, 0.98, 0.93])

    figure = DATA_ROOT / "exports/01_pool_daily_overview.png"
    fig.savefig(figure, dpi=220, bbox_inches="tight")
    plt.close(fig)

    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()

    manifest = {
        "source": "Uniswap V3 Subgraph via The Graph",
        "subgraph_id": SUBGRAPH,
        "git_commit": commit,
        "indexed_block": data["_meta"]["block"]["number"],
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_utc": START,
        "end_exclusive_utc": END,
        "row_count": len(df),
        "quality_checks": "PASS",
    }
    (DATA_ROOT / "_system/pool_daily_sep2026_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("Data quality: PASS")
    print("Rows:", len(df))
    print("Indexed block:", manifest["indexed_block"])
    print("Commit:", commit)
    print("Figure:", figure)

if __name__ == "__main__":
    main()
