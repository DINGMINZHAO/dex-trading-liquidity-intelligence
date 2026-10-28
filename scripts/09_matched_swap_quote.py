
import csv
import json
import os
from decimal import Decimal, getcontext
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_ROOT = Path(os.getenv("DEX_DATA_ROOT", ROOT / "data"))

BLOCK = 25889250
INPUT = (
    DATA_ROOT / "processed"
    / f"historical_pool_state_block_{BLOCK}.json"
)

OUTPUT_CSV = (
    DATA_ROOT / "analytics"
    / "matched_swap_quotes_block_25889250.csv"
)

OUTPUT_FIGURE = (
    ROOT / "reports/figures"
    / "06_matched_swap_quotes.png"
)

Q96 = 2**96
FEE_DENOMINATOR = 1_000_000

USDC_DECIMALS = 6
WETH_DECIMALS = 18

ORDER_SIZES_USDC = [
    1_000,
    5_000,
    10_000,
    20_000,
    50_000,
    100_000,
]

POOLS = {
    "0.05%": {
        "key": "low",
        "address": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
        "fee": 500,
        "tick_spacing": 10,
        "color": "#4A7BB7",
    },
    "0.30%": {
        "key": "high",
        "address": "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",
        "fee": 3000,
        "tick_spacing": 60,
        "color": "#F7834D",
    },
}

getcontext().prec = 90


def ceil_div(a, b):
    return (a + b - 1) // b


def price_usdc_per_weth(sqrt_x96):
    sqrt_p = Decimal(sqrt_x96) / Decimal(Q96)

    return (
        Decimal(10) ** (WETH_DECIMALS - USDC_DECIMALS)
        / (sqrt_p * sqrt_p)
    )


def tick_sqrt_x96(tick):
    sqrt_ratio = (
        Decimal("1.0001") ** tick
    ).sqrt()

    return int(sqrt_ratio * Decimal(Q96))


def load_states():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Historical pool state not found: {INPUT}"
        )

    payload = json.loads(INPUT.read_text())

    if payload["block_number"] != BLOCK:
        raise ValueError("Historical block mismatch")

    states = {}

    for label, config in POOLS.items():
        pool = payload["pools"][config["key"]]

        if pool["id"].lower() != config["address"]:
            raise ValueError(f"Pool address mismatch: {label}")

        if int(pool["feeTier"]) != config["fee"]:
            raise ValueError(f"Fee tier mismatch: {label}")

        token0 = pool["token0"]
        token1 = pool["token1"]

        if (
            token0["id"].lower()
            != "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
            or token1["id"].lower()
            != "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
            or int(token0["decimals"]) != USDC_DECIMALS
            or int(token1["decimals"]) != WETH_DECIMALS
        ):
            raise ValueError(f"Token metadata mismatch: {label}")

        liquidity = int(pool["liquidity"])
        sqrt_x96 = int(pool["sqrtPrice"])
        tick = int(pool["tick"])

        if liquidity <= 0 or sqrt_x96 <= 0:
            raise ValueError(f"Invalid pool state: {label}")

        spacing = config["tick_spacing"]

        lower_tick = (tick // spacing) * spacing
        upper_tick = lower_tick + spacing

        lower_sqrt = tick_sqrt_x96(lower_tick)
        upper_sqrt = tick_sqrt_x96(upper_tick)

        if not lower_sqrt <= sqrt_x96 <= upper_sqrt:
            raise ValueError(
                f"Tick and sqrtPrice inconsistent: {label}"
            )

        states[label] = {
            "liquidity": liquidity,
            "sqrt_x96": sqrt_x96,
            "tick": tick,
            "lower_tick": lower_tick,
            "lower_sqrt": lower_sqrt,
            "spot_price": price_usdc_per_weth(sqrt_x96),
        }

    return states


def simulate_exact_input_usdc(label, state, amount_usdc):
    config = POOLS[label]

    liquidity = state["liquidity"]
    current_sqrt = state["sqrt_x96"]

    amount_raw = amount_usdc * 10**USDC_DECIMALS

    fee_raw = ceil_div(
        amount_raw * config["fee"],
        FEE_DENOMINATOR,
    )

    net_input_raw = amount_raw - fee_raw

    # Uniswap V3 token0 exact-input price formula.
    numerator = (liquidity << 96) * current_sqrt
    denominator = (
        (liquidity << 96)
        + net_input_raw * current_sqrt
    )

    next_sqrt = ceil_div(numerator, denominator)

    base = {
        "block_number": BLOCK,
        "fee_tier": label,
        "input_usdc": amount_usdc,
        "initial_spot_usdc_per_weth": float(
            state["spot_price"]
        ),
        "lower_tick": state["lower_tick"],
    }

    # Do not extend constant liquidity beyond
    # the nearest possible initialized tick boundary.
    if next_sqrt < state["lower_sqrt"]:
        return {
            **base,
            "status": "SKIP_TICK_CROSSING",
            "weth_received": None,
            "execution_price": None,
            "execution_cost_bps": None,
            "terminal_spot_movement_bps": None,
        }

    # WETH output, rounded down in raw token units.
    weth_out_raw = (
        liquidity * (current_sqrt - next_sqrt)
    ) // Q96

    if weth_out_raw <= 0:
        return {
            **base,
            "status": "SKIP_ZERO_OUTPUT",
            "weth_received": None,
            "execution_price": None,
            "execution_cost_bps": None,
            "terminal_spot_movement_bps": None,
        }

    weth_received = (
        Decimal(weth_out_raw)
        / Decimal(10**WETH_DECIMALS)
    )

    execution_price = (
        Decimal(amount_usdc) / weth_received
    )

    initial_spot = state["spot_price"]
    final_spot = price_usdc_per_weth(next_sqrt)

    execution_cost_bps = (
        Decimal(10000)
        * (execution_price / initial_spot - 1)
    )

    terminal_movement_bps = (
        Decimal(10000)
        * (final_spot / initial_spot - 1)
    )

    return {
        **base,
        "status": "OK_SINGLE_RANGE",
        "weth_received": float(weth_received),
        "execution_price": float(execution_price),
        "execution_cost_bps": float(execution_cost_bps),
        "terminal_spot_movement_bps": float(
            terminal_movement_bps
        ),
    }


def create_figure(rows):
    OUTPUT_FIGURE.parent.mkdir(
        parents=True, exist_ok=True
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(11.5, 6.3))

    for label, config in POOLS.items():
        valid = sorted(
            (
                row for row in rows
                if row["fee_tier"] == label
                and row["status"] == "OK_SINGLE_RANGE"
            ),
            key=lambda x: x["input_usdc"],
        )

        ax.plot(
            [r["input_usdc"] for r in valid],
            [r["execution_price"] for r in valid],
            color=config["color"],
            marker="o",
            linewidth=2.2,
            markersize=7,
            label=label,
        )

    ax.set_xlabel("USDC input amount", labelpad=12)
    ax.set_ylabel(
        "Effective execution price (USDC/WETH)",
        labelpad=12,
    )

    ax.set_xticks(
        [1_000, 5_000, 10_000, 20_000]
    )

    ax.set_xticklabels(
        ["$1K", "$5K", "$10K", "$20K"]
    )

    ax.grid(
        axis="y",
        color="#E0E0E0",
        linewidth=0.8,
    )

    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B0B0B0")
    ax.spines["bottom"].set_color("#B0B0B0")
    ax.tick_params(length=0, colors="#333333", pad=8)

    ax.legend(
        title="Fee tier",
        frameon=False,
        loc="upper left",
    )

    fig.text(
        0.09, 0.95,
        "Uniswap V3 | Matched Swap Execution Quotes",
        fontsize=18,
        fontweight="bold",
        color="#1f1f1f",
    )

    fig.text(
        0.09, 0.905,
        f"USDC → WETH | Ethereum block {BLOCK} "
        "| Same input, same historical block",
        fontsize=10.5,
        color="#555555",
    )

    fig.text(
        0.09, 0.045,
        "Source: Uniswap V3 Subgraph / The Graph  |  "
        "Single-range model simulation; fees included",
        fontsize=8.5,
        color="#666666",
    )

    fig.text(
        0.09, 0.022,
        "Tick-crossing orders excluded. "
        "Gas, router fees and MEV excluded. "
        "Quotes are simulated, not executed.",
        fontsize=8.5,
        color="#666666",
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.97,
        top=0.82,
        bottom=0.19,
    )

    fig.savefig(
        OUTPUT_FIGURE,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    states = load_states()

    rows = []

    print(f"\n=== Historical Matched Quotes | Block {BLOCK} ===")

    for amount in ORDER_SIZES_USDC:
        for label in POOLS:
            rows.append(
                simulate_exact_input_usdc(
                    label, states[label], amount
                )
            )

    OUTPUT_CSV.parent.mkdir(
        parents=True, exist_ok=True
    )

    with OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Simulated USDC → WETH Quotes ===")

    for amount in ORDER_SIZES_USDC:
        print(f"\nInput: {amount:,} USDC")

        results = {}

        for row in rows:
            if row["input_usdc"] != amount:
                continue

            label = row["fee_tier"]
            results[label] = row

            if row["status"] != "OK_SINGLE_RANGE":
                print(f"  {label}: {row['status']}")
                continue

            print(
                f"  {label}: "
                f"{row['weth_received']:.9f} WETH | "
                f"Effective price ${row['execution_price']:.4f} "
                f"| Cost {row['execution_cost_bps']:.3f} bps"
            )

        if all(
            results[tier]["status"] == "OK_SINGLE_RANGE"
            for tier in POOLS
        ):
            low = results["0.05%"]
            high = results["0.30%"]

            winner = (
                "0.05%"
                if low["weth_received"] > high["weth_received"]
                else "0.30%"
            )

            difference = abs(
                low["weth_received"]
                - high["weth_received"]
            )

            print(
                f"  Better quote: {winner} | "
                f"WETH difference: {difference:.9f}"
            )

    create_figure(rows)

    print("\n=== Output ===")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Figure: {OUTPUT_FIGURE}")
    print("Simulation complete.")


if __name__ == "__main__":
    main()
