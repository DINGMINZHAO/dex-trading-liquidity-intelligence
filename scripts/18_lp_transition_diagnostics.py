
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA = Path(os.getenv("DEX_DATA_ROOT", ROOT / "data"))
SOURCE = DATA / "processed/q3_lp_activity"
ANALYTICS = DATA / "analytics"
FIGURES = ROOT / "reports/figures"

START = pd.Timestamp("2026-08-10", tz="UTC")
END = pd.Timestamp("2026-09-11", tz="UTC")
TRANSITION = pd.Timestamp("2026-08-17", tz="UTC")
POST_START = pd.Timestamp("2026-08-24", tz="UTC")

BLUE = "#4A7BB7"
ORANGE = "#F7834D"
RED = "#D9412B"


def load_data():
    daily = pd.read_csv(
        ANALYTICS / "q3_active_liquidity_daily.csv"
    )
    daily["date"] = pd.to_datetime(daily["date"], utc=True)

    daily = daily[
        (daily["fee_tier"] == "0.30%")
        & (daily["date"] >= START)
        & (daily["date"] < END)
    ].sort_values("date").copy()

    expected = pd.date_range(
        START, END, freq="D", inclusive="left"
    )

    if (
        len(daily) != len(expected)
        or not daily["date"].reset_index(drop=True).equals(
            pd.Series(expected, name="date")
        )
    ):
        raise ValueError("Incomplete daily data")

    daily["tick"] = pd.to_numeric(
        daily["tick"], errors="coerce"
    )

    required = [
        "tick", "active_liquidity", "tvl_usd",
        "volume_share_pct"
    ]

    if not np.isfinite(
        daily[required].to_numpy(dtype=float)
    ).all():
        raise ValueError("Invalid daily state data")

    frames = []

    for kind in ["mints", "burns"]:
        path = SOURCE / f"{kind}_3000.parquet"

        if not path.exists():
            raise FileNotFoundError(path)

        part = pd.read_parquet(path)
        part["event_type"] = kind
        frames.append(part)

    events = pd.concat(frames, ignore_index=True)

    if events["event_id"].duplicated().any():
        raise ValueError("Duplicate LP event IDs")

    events["date"] = pd.to_datetime(
        events["date"], utc=True
    )

    if not events["date"].between(
        START, END, inclusive="left"
    ).all():
        raise ValueError("LP event outside focus period")

    events = events[
        events["nonzero_liquidity"] == True
    ].copy()

    events["tick_lower"] = pd.to_numeric(
        events["tick_lower"]
    )
    events["tick_upper"] = pd.to_numeric(
        events["tick_upper"]
    )

    daily_ticks = daily.set_index("date")["tick"]

    events["day_end_tick"] = (
        events["date"].map(daily_ticks)
    )

    events["range_covers_day_end_tick"] = (
        (events["tick_lower"] <= events["day_end_tick"])
        & (events["day_end_tick"] < events["tick_upper"])
    )

    return daily, events


def calculate(daily, events):
    df = daily.copy()

    pre = df[
        (df["date"] >= START)
        & (df["date"] < TRANSITION)
    ]

    for field in ["active_liquidity", "tvl_usd"]:
        baseline = pre[field].median()

        if baseline <= 0:
            raise ValueError(f"Invalid baseline: {field}")

        df[f"{field}_index"] = (
            100 * df[field] / baseline
        )

    df["active_change_pct"] = (
        df["active_liquidity"].pct_change() * 100
    )

    df["tick_change"] = df["tick"].diff()

    grouped = events.groupby(
        ["date", "event_type"]
    ).agg(
        event_count=("event_id", "size"),
        covers_day_end_tick=(
            "range_covers_day_end_tick", "sum"
        )
    )

    for kind in ["mints", "burns"]:
        part = grouped.xs(
            kind, level="event_type"
        ) if kind in grouped.index.get_level_values(
            "event_type"
        ) else pd.DataFrame()

        for source, target in [
            ("event_count", f"{kind}_events"),
            ("covers_day_end_tick", f"{kind}_day_end_in_range"),
        ]:
            if part.empty:
                df[target] = 0
            else:
                df[target] = (
                    df["date"].map(part[source])
                    .fillna(0).astype(int)
                )

    df["net_event_count"] = (
        df["mints_events"] - df["burns_events"]
    )

    return df


def plot(df):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, axes = plt.subplots(
        2, 2, figsize=(15, 9.5), sharex=True
    )

    ax = axes[0, 0]
    ax.plot(
        df["date"], df["active_liquidity_index"],
        color=ORANGE, linewidth=2.2,
        label="Active liquidity"
    )
    ax.plot(
        df["date"], df["tvl_usd_index"],
        color=BLUE, linewidth=2.2,
        label="TVL"
    )
    ax.set_title(
        "A | Active Liquidity vs TVL",
        loc="left", fontweight="bold"
    )
    ax.set_ylabel("Index (Aug 10–16 median = 100)")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(
        df["date"], df["tick"],
        color=ORANGE, marker="o",
        markersize=3, linewidth=1.7
    )
    ax.set_title(
        "B | Daily Closing Tick",
        loc="left", fontweight="bold"
    )
    ax.set_ylabel("Uniswap V3 tick")

    ax = axes[1, 0]
    x = mdates.date2num(df["date"])

    ax.bar(
        x - 0.2, df["mints_events"],
        width=0.4, color=BLUE, label="Mint"
    )
    ax.bar(
        x + 0.2, df["burns_events"],
        width=0.4, color=RED, label="Burn"
    )
    ax.set_title(
        "C | Nonzero Mint/Burn Activity",
        loc="left", fontweight="bold"
    )
    ax.set_ylabel("Events per day")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(
        df["date"], df["volume_share_pct"],
        color=ORANGE, linewidth=2,
        marker="o", markersize=3
    )
    ax.set_title(
        "D | 0.30% Pool Trading Volume Share",
        loc="left", fontweight="bold"
    )
    ax.set_ylabel("Share of two pools (%)")

    for ax in axes.flat:
        ax.axvspan(
            TRANSITION, POST_START,
            color="#FED081", alpha=0.18
        )
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(
            mdates.WeekdayLocator(
                byweekday=mdates.MO, tz=mdates.UTC
            )
        )
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter(
                "%b %d", tz=mdates.UTC
            )
        )
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle(
        "Uniswap V3 | 0.30% Pool Liquidity Transition",
        fontsize=19, fontweight="bold", y=0.99
    )

    fig.text(
        0.5, 0.945,
        "USDC/WETH | Aug 10 – Sep 10, 2026 (UTC)"
        " | Shaded: Aug 17–23",
        ha="center", color="#555555"
    )

    fig.text(
        0.5, 0.018,
        "Source: The Graph | Daily closing states"
        " | Mint/Burn counts do not measure net capital flow",
        ha="center", fontsize=9, color="#666666"
    )

    fig.tight_layout(
        rect=[0.025, 0.05, 0.99, 0.91],
        h_pad=2.5, w_pad=2.3
    )

    FIGURES.mkdir(parents=True, exist_ok=True)

    path = (
        FIGURES / "15_q3_lp_transition_diagnostics.png"
    )

    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    daily, events = load_data()
    result = calculate(daily, events)

    ANALYTICS.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        ANALYTICS / "q3_lp_transition_diagnostics.csv",
        index=False
    )

    print("\n=== LARGEST DAILY ACTIVE LIQUIDITY CHANGES ===")

    top = (
        result.assign(
            abs_change=result["active_change_pct"].abs()
        )
        .nlargest(10, "abs_change")
        .sort_values("date")
    )

    cols = [
        "date", "active_change_pct", "tick_change",
        "mints_events", "burns_events",
        "mints_day_end_in_range",
        "burns_day_end_in_range"
    ]

    print(
        top[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:,.2f}"
        )
    )

    print("\n=== BEFORE / TRANSITION / AFTER ===")

    windows = [
        ("Before", "2026-08-10", "2026-08-17"),
        ("Transition", "2026-08-17", "2026-08-24"),
        ("After", "2026-08-24", "2026-08-31"),
    ]

    for name, start, end in windows:
        part = result[
            (result["date"] >= pd.Timestamp(start, tz="UTC"))
            & (result["date"] < pd.Timestamp(end, tz="UTC"))
        ]

        print(
            f"{name}: "
            f"Active index={part['active_liquidity_index'].median():.1f}, "
            f"TVL index={part['tvl_usd_index'].median():.1f}, "
            f"Volume share={part['volume_share_pct'].median():.2f}%, "
            f"Mint={part['mints_events'].sum()}, "
            f"Burn={part['burns_events'].sum()}"
        )

    print("\nNote: In-range event counts use the daily closing")
    print("tick, not the tick at the event's actual block.")

    figure = plot(result)
    print(f"\nFigure: {figure}")
    print("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
