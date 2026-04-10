"""
Generate charts from main/outputs/*.csv aligned with leaderboard evaluation themes:
  1) overall forecast (point trajectory)
  2) high-load / risk (p90 uplift and peak-hour proxy)
  3) upper-risk estimate quality (risk gap distribution)

Run from repo root:
  python3 -m main.src.reporting.visualize_outputs
  python3 -m main.src.reporting.visualize_outputs --out-dir /path/to/main/outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _load(out_dir: Path, name: str) -> pd.DataFrame:
    p = out_dir / name
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def run(out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    pred = _load(out_dir, "prediction_24h.csv")
    pred["timestamp_utc"] = pd.to_datetime(pred["timestamp_utc"])
    pred["risk_gap_kw"] = pred["pred_p90_kw"] - pred["pred_power_kw"]
    peak_thr = pred["pred_power_kw"].quantile(0.9)
    pred["is_peak_proxy"] = pred["pred_power_kw"] >= peak_thr

    # --- Fig 1–3: evaluation-aligned forecast view ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(pred["timestamp_utc"], pred["pred_power_kw"], label="pred_power_kw (point)", color="#2a9d8f")
    axes[0].plot(pred["timestamp_utc"], pred["pred_p90_kw"], label="pred_p90_kw (upper risk)", color="#e76f51", alpha=0.85)
    axes[0].set_ylabel("kW")
    axes[0].set_title("1) Overall forecast + 3) P90 upper estimate (leaderboard criteria 1 & 3)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(pred["timestamp_utc"], pred["risk_gap_kw"], width=0.03, color="#457b9d", alpha=0.7)
    axes[1].set_ylabel("kW")
    axes[1].set_title("Risk gap (pred_p90 − pred_power): higher gap = more conservative upper bound")
    axes[1].grid(True, alpha=0.3)

    peak = pred[pred["is_peak_proxy"]]
    axes[2].scatter(
        pred["timestamp_utc"],
        pred["pred_power_kw"],
        c=pred["is_peak_proxy"].map({True: "#e63946", False: "#adb5bd"}),
        s=12,
        alpha=0.8,
    )
    axes[2].set_ylabel("kW")
    axes[2].set_title(f"2) Peak-hour proxy: hours ≥ 90th pct of point forecast (n={len(peak)})")
    axes[2].grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(fig_dir / "01_evaluation_forecast_peak_p90.png", dpi=150)
    plt.close()

    # Breakdown columns if present
    if {"hardware_contribution", "ambient_weather_adj", "tier_size_adj"}.issubset(pred.columns):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(pred["timestamp_utc"], pred["hardware_contribution"], label="hardware (lag kW)")
        ax.plot(pred["timestamp_utc"], pred["ambient_weather_adj"], label="ambient adj (kW)")
        ax.plot(pred["timestamp_utc"], pred["tier_size_adj"], label="tier mix adj (kW)")
        ax.set_ylabel("kW")
        ax.legend()
        ax.set_title("Interpretation helpers (from lag-hour mix; not additive to pred)")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(fig_dir / "02_prediction_breakdown.png", dpi=150)
        plt.close()

    # Hardware
    hw = _load(out_dir, "analysis_hardware.csv")
    top = hw.nlargest(12, "mean")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top["HardwareType"].astype(str), top["mean"], color="#264653")
    ax.invert_yaxis()
    ax.set_xlabel("Mean power (W, per row)")
    ax.set_title("Hardware / brand — mean consumption")
    plt.tight_layout()
    fig.savefig(fig_dir / "03_hardware_mean.png", dpi=150)
    plt.close()

    # Ambient
    amb = _load(out_dir, "analysis_ambient.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(amb["ambient_band"].astype(str), amb["mean"], color="#2a9d8f")
    ax.set_ylabel("Mean power (W)")
    ax.set_title("Ambient temperature bands vs mean power")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(fig_dir / "04_ambient_bands.png", dpi=150)
    plt.close()

    # Weather (may be fewer rows after empty-band drop)
    wpath = out_dir / "analysis_weather_3mo.csv"
    if wpath.exists():
        w = pd.read_csv(wpath)
        w = w.dropna(subset=["count"])
        if len(w):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(w["weather_band"].astype(str), w["mean"], color="#e9c46a")
            ax.set_ylabel("Mean terminal power (kW) in merge window")
            ax.set_title("Weather temperature bands vs hourly terminal load (merged period)")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            fig.savefig(fig_dir / "05_weather_bands.png", dpi=150)
            plt.close()

    # Tier x size heatmap-style
    tsz = _load(out_dir, "analysis_tier_size.csv")
    pivot = tsz.pivot_table(index="stack_tier", columns="ContainerSize", values="mean", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Mean power (W) — tier × container size")
    plt.colorbar(im, ax=ax, label="W")
    plt.tight_layout()
    fig.savefig(fig_dir / "06_tier_size_heatmap.png", dpi=150)
    plt.close()

    print(f"Saved figures under {fig_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "outputs",
        help="Directory containing analysis_*.csv and prediction_24h.csv",
    )
    args = parser.parse_args()
    run(args.out_dir)


if __name__ == "__main__":
    main()
