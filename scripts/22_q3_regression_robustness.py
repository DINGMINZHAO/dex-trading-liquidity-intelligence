
import math
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
ANALYTICS = DATA / "analytics"
FIGURES = ROOT / "reports/figures"

BLUE = "#4A7BB7"
ORANGE = "#F7834D"
RED = "#D9412B"

MONTH_COLORS = {
    "2026-07": BLUE,
    "2026-08": ORANGE,
    "2026-09": RED,
}


def prepare():
    vol = pd.read_csv(
        ANALYTICS / "q3_volatility_regimes_daily.csv"
    )
    liq = pd.read_csv(
        ANALYTICS / "q3_liquidity_daily.csv"
    )
    active = pd.read_csv(
        ANALYTICS / "q3_active_liquidity_daily.csv"
    )

    for df in [vol, liq, active]:
        df["date"] = pd.to_datetime(df["date"], utc=True)

    high = liq[liq["fee_tier"] == "0.30%"][
        ["date", "volume_share_pct", "tvl_usd"]
    ].rename(columns={
        "volume_share_pct": "share_pct",
        "tvl_usd": "high_tvl",
    })

    low = liq[liq["fee_tier"] == "0.05%"][
        ["date", "tvl_usd"]
    ].rename(columns={"tvl_usd": "low_tvl"})

    active_high = active[
        active["fee_tier"] == "0.30%"
    ][["date", "active_liquidity_index"]]

    df = (
        vol[["date", "realized_vol_pct", "regime"]]
        .merge(high, on="date", validate="one_to_one")
        .merge(low, on="date", validate="one_to_one")
        .merge(active_high, on="date", validate="one_to_one")
        .sort_values("date")
        .reset_index(drop=True)
    )

    expected = pd.date_range(
        "2026-07-01", "2026-09-30", tz="UTC"
    )

    if (
        len(df) != 92
        or not df["date"].equals(
            pd.Series(expected, name="date")
        )
    ):
        raise ValueError("Incomplete Q3 daily observations")

    required = [
        "realized_vol_pct", "share_pct",
        "high_tvl", "low_tvl",
        "active_liquidity_index",
    ]

    if not np.isfinite(df[required].to_numpy()).all():
        raise ValueError("Non-finite observations")

    if (
        (df["share_pct"] <= 0).any()
        or (df["share_pct"] >= 100).any()
        or (df["high_tvl"] <= 0).any()
        or (df["low_tvl"] <= 0).any()
        or (df["active_liquidity_index"] <= 0).any()
    ):
        raise ValueError("Invalid share or liquidity data")

    df["month"] = df["date"].dt.strftime("%Y-%m")
    df["rv_pct"] = df["realized_vol_pct"]
    df["aug"] = (df["month"] == "2026-08").astype(float)
    df["sep"] = (df["month"] == "2026-09").astype(float)

    df["log_tvl_ratio"] = np.log(
        df["high_tvl"] / df["low_tvl"]
    )
    df["log_active_index"] = np.log(
        df["active_liquidity_index"] / 100
    )

    share_fraction = df["share_pct"] / 100
    df["share_logodds"] = np.log(
        share_fraction / (1 - share_fraction)
    )

    return df


def fit_hac(df, name, predictors, target="share_pct", lags=3):
    y = df[target].to_numpy(dtype=float)

    x = np.column_stack(
        [np.ones(len(df))]
        + [df[col].to_numpy(dtype=float) for col in predictors]
    )

    n, p = x.shape

    if n <= p or np.linalg.matrix_rank(x) != p:
        raise ValueError(f"Invalid regression design: {name}")

    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ beta

    inverse = np.linalg.inv(x.T @ x)
    scores = x * residuals[:, None]

    meat = scores.T @ scores

    for lag in range(1, min(lags, n - 1) + 1):
        gamma = scores[lag:].T @ scores[:-lag]
        weight = 1 - lag / (lags + 1)
        meat += weight * (gamma + gamma.T)

    covariance = inverse @ meat @ inverse
    standard_errors = np.sqrt(
        np.maximum(np.diag(covariance), 0)
    )

    # rv_pct is always the first predictor.
    coefficient = float(beta[1])
    se = float(standard_errors[1])

    z = coefficient / se if se > 0 else np.nan
    pvalue = (
        math.erfc(abs(z) / math.sqrt(2))
        if np.isfinite(z) else np.nan
    )

    total = np.square(y - y.mean()).sum()
    rss = np.square(residuals).sum()
    r2 = 1 - rss / total if total > 0 else np.nan

    return {
        "model": name,
        "outcome": target,
        "observations": n,
        "vol_coefficient": coefficient,
        "hac_se": se,
        "ci_lower": coefficient - 1.96 * se,
        "ci_upper": coefficient + 1.96 * se,
        "p_value_approx": pvalue,
        "r_squared": r2,
        "predictors": ", ".join(predictors),
    }


def run_models(df):
    month = ["rv_pct", "aug", "sep"]
    tvl = month + ["log_tvl_ratio"]
    full = tvl + ["log_active_index"]

    transition = (
        (df["date"] >= pd.Timestamp("2026-08-18", tz="UTC"))
        & (df["date"] < pd.Timestamp("2026-08-24", tz="UTC"))
    )

    extreme_dates = set(
        df.nlargest(2, "rv_pct")["date"]
    )

    results = [
        fit_hac(df, "M1: Month FE", month),
        fit_hac(df, "M2: Month FE + TVL", tvl),
        fit_hac(df, "M3: + Active liquidity", full),
        fit_hac(
            df.loc[~transition],
            "R1: Exclude Aug transition",
            tvl,
        ),
        fit_hac(
            df.loc[~df["date"].isin(extreme_dates)],
            "R2: Exclude top 2 RV days",
            tvl,
        ),
        fit_hac(
            df,
            "R3: Log-odds share",
            tvl,
            target="share_logodds",
        ),
    ]

    return pd.DataFrame(results)


def plot_results(df, models):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5))

    ax = axes[0, 0]

    for month, color in MONTH_COLORS.items():
        part = df[df["month"] == month]

        ax.scatter(
            part["rv_pct"],
            part["share_pct"],
            color=color,
            s=43,
            alpha=0.8,
            label=month,
        )

    ax.set_title(
        "A | Volatility vs 0.30% Volume Share",
        loc="left", fontweight="bold"
    )
    ax.set_xlabel("Daily realized volatility (%)")
    ax.set_ylabel("0.30% Pool volume share (%)")
    ax.legend(frameon=False)

    ax = axes[0, 1]

    centered = df.copy()

    for col in ["rv_pct", "share_pct"]:
        centered[f"{col}_centered"] = (
            centered[col]
            - centered.groupby("month")[col].transform("mean")
        )

    for month, color in MONTH_COLORS.items():
        part = centered[centered["month"] == month]

        ax.scatter(
            part["rv_pct_centered"],
            part["share_pct_centered"],
            color=color,
            alpha=0.7,
            s=40,
        )

    slope = models.iloc[0]["vol_coefficient"]

    xx = np.linspace(
        centered["rv_pct_centered"].min(),
        centered["rv_pct_centered"].max(),
        100
    )

    ax.plot(xx, slope * xx, color="#333333", linewidth=1.6)

    ax.set_title(
        "B | Within-Month Association",
        loc="left", fontweight="bold"
    )
    ax.set_xlabel("Volatility deviation from monthly mean (pp)")
    ax.set_ylabel("Volume share deviation (pp)")

    ax = axes[1, 0]

    selected = models[
        models["outcome"] == "share_pct"
    ].iloc[::-1]

    y = np.arange(len(selected))

    ax.errorbar(
        selected["vol_coefficient"],
        y,
        xerr=[
            selected["vol_coefficient"] - selected["ci_lower"],
            selected["ci_upper"] - selected["vol_coefficient"],
        ],
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        capsize=4,
    )

    ax.axvline(
        0, color="#888888", linestyle="--", linewidth=1
    )

    ax.set_yticks(y, selected["model"])
    ax.set_xlabel(
        "Volume share change (pp) per +1 pp daily RV"
    )
    ax.set_title(
        "C | Regression Robustness (HAC 95% CI)",
        loc="left", fontweight="bold"
    )

    ax = axes[1, 1]

    ax.plot(
        df["date"],
        df["share_pct"].rolling(7, min_periods=1).mean(),
        color=ORANGE,
        linewidth=2,
    )

    ax.axvspan(
        pd.Timestamp("2026-08-18", tz="UTC"),
        pd.Timestamp("2026-08-24", tz="UTC"),
        color="#FED081",
        alpha=0.35,
    )

    ax.set_title(
        "D | 0.30% Pool Share Over Time",
        loc="left", fontweight="bold"
    )
    ax.set_ylabel("7-day moving average (%)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%b %Y")
    )

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Uniswap V3 | Volatility and Market Share Robustness",
        fontsize=19, fontweight="bold", y=0.99
    )

    fig.text(
        0.5, 0.945,
        "USDC/WETH | Q3 2026 | Daily observations",
        ha="center", color="#555555"
    )

    fig.text(
        0.5, 0.017,
        "HAC standard errors: Newey-West, 3 lags | "
        "Month fixed effects | Observational associations",
        ha="center", fontsize=9, color="#666666"
    )

    fig.tight_layout(
        rect=[0.035, 0.055, 0.99, 0.91],
        h_pad=3, w_pad=3
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    output = FIGURES / "19_q3_regression_robustness.png"

    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return output


def main():
    ANALYTICS.mkdir(parents=True, exist_ok=True)

    df = prepare()
    models = run_models(df)

    df.to_csv(
        ANALYTICS / "q3_regression_daily_dataset.csv",
        index=False
    )

    models.to_csv(
        ANALYTICS / "q3_regression_robustness.csv",
        index=False
    )

    figure = plot_results(df, models)

    print("\n=== REGRESSION & ROBUSTNESS ===")

    columns = [
        "model", "observations",
        "vol_coefficient", "hac_se",
        "ci_lower", "ci_upper",
        "p_value_approx", "r_squared"
    ]

    print(
        models[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:,.4f}"
        )
    )

    print("\nInterpretation:")
    print("M1-M3/R1-R2: pp of volume share per +1 pp RV.")
    print("R3: log-odds change; not directly comparable.")
    print("HAC p-values use large-sample normal approximation.")
    print("Associations do not establish causality.")

    print(f"\nFigure: {figure}")
    print("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
