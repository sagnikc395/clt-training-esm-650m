#!/usr/bin/env python3
"""
Analyze PLT sweep results from wandb.

Fetches finished sweep runs, aggregates per-layer training and validation metrics
across the 4 GPU shards, and ranks configurations using validation faithfulness
metrics when available.

Usage:
    python analyze_sweep.py
    python analyze_sweep.py --project user/sparsify
    python analyze_sweep.py --export results.csv
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

PREFIX_METRICS = {
    "fvu/": "mean_fvu",
    "l0/": "mean_l0",
    "dead_pct/": "mean_dead_pct",
    "density_mean_ema/": "mean_density_mean_ema",
    "dense_pct_ema/": "mean_dense_pct_ema",
    "val_fvu/": "val_mean_fvu",
    "val_l0/": "val_mean_l0",
    "val_dead_pct/": "val_mean_dead_pct",
    "val_density_mean/": "val_mean_density_mean",
    "val_dense_pct/": "val_mean_dense_pct",
}

SCALAR_METRICS = [
    "ce_loss",
    "kl_loss",
    "acc_top1",
    "mean_fvu",
    "mean_l0",
    "mean_dead_pct",
    "mean_density_mean_ema",
    "mean_dense_pct_ema",
    "val_mean_fvu",
    "val_mean_l0",
    "val_mean_dead_pct",
    "val_mean_density_mean",
    "val_mean_dense_pct",
    "val_replacement_ce_clean",
    "val_replacement_ce_dirty",
    "val_replacement_ce_increase",
    "val_replacement_kl",
    "val_replacement_acc_top1",
]


def safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean_or_none(values):
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if v is not None and pd.notna(v)]
    return sum(vals) / len(vals) if vals else None


def parse_hparams_from_name(name: str) -> dict[str, object]:
    match = re.search(
        r"ef(?P<expansion_factor>\d+)_k(?P<k>\d+)_lr(?P<lr>[^_]+)_opt(?P<optimizer>[^_]+)",
        name,
    )
    if match is None:
        return {}
    return {
        "lr": match.group("lr"),
        "k": int(match.group("k")),
        "expansion_factor": int(match.group("expansion_factor")),
        "optimizer": match.group("optimizer"),
    }


def fetch_sweep_runs(project: str, sweep_prefix: str = r"(?:sweep_|optuna_)"):
    """Fetch finished sweep runs from wandb."""
    import wandb

    api = wandb.Api()
    runs = api.runs(project, filters={"display_name": {"$regex": f"^{sweep_prefix}"}})

    results = []
    for run in runs:
        if run.state != "finished":
            continue

        name = run.name
        config = run.config
        summary = run.summary

        # Extract hyperparams from config
        hp = {
            "run_name": name,
            "lr": config.get("lr"),
            "k": config.get("k"),
            "expansion_factor": config.get("expansion_factor"),
            "optimizer": config.get("optimizer"),
            "batch_size": config.get("batch_size"),
            "state": run.state,
        }
        hp.update(
            {
                k: v
                for k, v in parse_hparams_from_name(name).items()
                if hp.get(k) is None
            }
        )

        layer_metrics = {}
        for key, val in summary.items():
            for prefix in PREFIX_METRICS:
                if key.startswith(prefix):
                    layer_metrics[key] = safe_float(val)
                    break

        for metric in SCALAR_METRICS:
            hp[metric] = safe_float(summary.get(metric))

        train_fvus = [
            v
            for k, v in layer_metrics.items()
            if k.startswith("fvu/") and v is not None
        ]
        val_fvus = [
            v
            for k, v in layer_metrics.items()
            if k.startswith("val_fvu/") and v is not None
        ]
        hp["mean_fvu"] = (
            hp["mean_fvu"] if hp["mean_fvu"] is not None else mean_or_none(train_fvus)
        )
        hp["val_mean_fvu"] = (
            hp["val_mean_fvu"]
            if hp["val_mean_fvu"] is not None
            else mean_or_none(val_fvus)
        )
        hp["max_fvu"] = max(train_fvus) if train_fvus else None
        hp["min_fvu"] = min(train_fvus) if train_fvus else None
        hp["num_layers"] = len(train_fvus)
        hp["num_val_layers"] = len(val_fvus)
        hp.update(layer_metrics)

        results.append(hp)

    return pd.DataFrame(results)


def group_by_config(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group GPU-specific runs (e.g. sweep_..._g0, sweep_..._g1) into
    a single row per hyperparameter configuration.
    """
    # Strip _g0, _g1, etc. from run names to get the base config name
    df = df.copy()
    df["config_name"] = df["run_name"].apply(
        lambda x: re.sub(r"_g\d+$", "", x) if isinstance(x, str) else x
    )

    hp_cols = ["lr", "k", "expansion_factor", "optimizer", "batch_size"]
    scalar_cols = [c for c in SCALAR_METRICS if c in df.columns]

    grouped = []
    for config_name, group in df.groupby("config_name"):
        row = {col: group[col].iloc[0] for col in hp_cols}
        row["config_name"] = config_name
        row["num_gpu_runs"] = len(group)

        for col in scalar_cols:
            row[col] = mean_or_none(group[col].tolist())

        for prefix, aggregate_name in PREFIX_METRICS.items():
            metric_cols = [c for c in group.columns if c.startswith(prefix)]
            values = []
            for col in metric_cols:
                vals = group[col].dropna()
                if len(vals) > 0:
                    values.extend(vals.tolist())
            if values:
                row[aggregate_name] = mean_or_none(values)
                row[f"{aggregate_name}_count"] = len(values)

        train_fvus = []
        for col in [c for c in group.columns if c.startswith("fvu/")]:
            train_fvus.extend(group[col].dropna().tolist())
        row["mean_fvu"] = (
            row.get("mean_fvu")
            if row.get("mean_fvu") is not None
            else mean_or_none(train_fvus)
        )
        row["max_fvu"] = max(train_fvus) if train_fvus else None
        row["min_fvu"] = min(train_fvus) if train_fvus else None
        row["num_layers"] = len(train_fvus)

        val_fvus = []
        for col in [c for c in group.columns if c.startswith("val_fvu/")]:
            val_fvus.extend(group[col].dropna().tolist())
        row["num_val_layers"] = len(val_fvus)
        grouped.append(row)

    grouped_df = pd.DataFrame(grouped)
    grouped_df["sort_primary"] = grouped_df["val_mean_fvu"].where(
        grouped_df["val_mean_fvu"].notna(), grouped_df["mean_fvu"]
    )
    grouped_df["sort_secondary"] = grouped_df["val_replacement_ce_increase"].where(
        grouped_df["val_replacement_ce_increase"].notna(), float("inf")
    )
    grouped_df["sort_tertiary"] = grouped_df["val_mean_l0"].where(
        grouped_df["val_mean_l0"].notna(), grouped_df["mean_l0"]
    )
    return grouped_df.sort_values(
        ["sort_primary", "sort_secondary", "sort_tertiary"],
        ascending=[True, True, True],
        na_position="last",
    )


def map_configs_to_indices(top_configs: pd.DataFrame, config_path: str) -> list[int]:
    """Map top configs (by hp values) back to their combo index in the sweep grid."""
    import itertools
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    sp = cfg["sweep_params"]
    keys = sorted(sp.keys())
    vals = [sp[k] for k in keys]
    # Build index lookup: (ef, k, lr, optimizer) -> index
    combo_to_idx = {}
    for i, combo in enumerate(itertools.product(*vals)):
        # Normalize: convert all values to strings for matching
        combo_key = tuple(str(v) for v in combo)
        combo_to_idx[combo_key] = i

    indices = []
    for _, row in top_configs.iterrows():
        key = tuple(str(row.get(k, "")) for k in keys)
        idx = combo_to_idx.get(key)
        if idx is not None:
            indices.append(idx)
    return indices


def main():
    parser = argparse.ArgumentParser(description="Analyze PLT sweep results")
    parser.add_argument(
        "--project",
        default="cricksai6-university-of-massachusetts-amherst/sparsify",
        help="wandb project path (entity/project)",
    )
    parser.add_argument(
        "--prefix",
        default=r"(?:sweep_|optuna_)",
        help="Regex prefix used to filter wandb display names",
    )
    parser.add_argument(
        "--export",
        default=None,
        help="Export results to CSV file",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Number of top configs to select",
    )
    parser.add_argument(
        "--output-indices",
        default=None,
        help="Write top-N combo indices as JSON to this file (for automation)",
    )
    args = parser.parse_args()

    print(f"Fetching sweep runs from {args.project}...")
    df = fetch_sweep_runs(args.project, args.prefix)

    if df.empty:
        print("No completed sweep runs found.")
        return

    print(f"Found {len(df)} completed GPU-level runs")

    # Group by config (aggregate across GPUs)
    grouped = group_by_config(df)
    print(f"Grouped into {len(grouped)} unique configurations\n")

    # Display top results
    display_cols = [
        "config_name",
        "lr",
        "k",
        "expansion_factor",
        "optimizer",
        "mean_fvu",
        "val_mean_fvu",
        "val_replacement_ce_increase",
        "val_replacement_kl",
        "val_replacement_acc_top1",
        "mean_l0",
        "val_mean_l0",
        "mean_dead_pct",
        "val_mean_dead_pct",
        "max_fvu",
        "num_layers",
        "num_val_layers",
    ]
    available = [c for c in display_cols if c in grouped.columns]

    print("=" * 80)
    print("TOP CONFIGURATIONS (sorted by validation FVU, then replacement CE increase)")
    print("=" * 80)
    print(grouped[available].head(20).to_string(index=False))
    print()

    # Best config
    best = grouped.iloc[0]
    print("=" * 80)
    print("BEST CONFIGURATION:")
    print(
        f"  lr={best['lr']}, k={best['k']}, "
        f"expansion_factor={best['expansion_factor']}, "
        f"optimizer={best['optimizer']}"
    )
    primary_name = "val_mean_fvu" if pd.notna(best.get("val_mean_fvu")) else "mean_fvu"
    print(f"  Primary metric ({primary_name}): {best[primary_name]:.6f}")
    if pd.notna(best.get("val_replacement_ce_increase")):
        print(
            f"  Validation replacement CE increase: {best['val_replacement_ce_increase']:.6f}"
        )
    print("=" * 80)

    if args.export:
        grouped.to_csv(args.export, index=False)
        print(f"\nResults exported to {args.export}")

    # Output top-N indices for automation
    top_n = args.top_n or len(grouped)
    top_configs = grouped.head(top_n)
    config_path = Path(__file__).resolve().parent / "sweep_config.yaml"
    indices = map_configs_to_indices(top_configs, str(config_path))
    if args.top_n:
        print(f"\nTop {top_n} combo indices: {indices}")
    if args.output_indices:
        with open(args.output_indices, "w") as f:
            json.dump(indices, f)
        print(f"Indices written to {args.output_indices}")


if __name__ == "__main__":
    main()
