"""Utility functions for irregular time-series dataset exploration notebooks."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


@dataclass
class IrregularDatasetBundle:
    name: str
    records: list[dict]
    channel_names: list[str]
    time_unit: str
    description: str


def _record_to_long_frame(
    sample_id: int,
    rec: dict,
    channel_names_arr: np.ndarray,
    n_channels: int,
) -> pd.DataFrame | None:
    t = np.asarray(rec["t"], dtype=float)
    x = np.asarray(rec["x"], dtype=float)
    if x.ndim != 2 or x.shape[1] != n_channels:
        return None

    finite_mask = np.isfinite(x)
    if not finite_mask.any():
        return None

    time_idx, chan_idx = np.where(finite_mask)
    return pd.DataFrame(
        {
            "sample_id": np.full(time_idx.shape[0], sample_id, dtype=np.int64),
            "time": t[time_idx],
            "channel": channel_names_arr[chan_idx],
            "value": x[time_idx, chan_idx],
        }
    )


def _to_long_frame_impl(records: list[dict], channel_names: list[str], n_jobs: int = 1) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    n_channels = len(channel_names)
    channel_names_arr = np.asarray(channel_names, dtype=object)

    if n_jobs <= 1:
        for sample_id, rec in enumerate(records):
            df = _record_to_long_frame(sample_id, rec, channel_names_arr, n_channels)
            if df is not None:
                rows.append(df)
    else:
        max_workers = min(n_jobs, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_record_to_long_frame, sample_id, rec, channel_names_arr, n_channels)
                for sample_id, rec in enumerate(records)
            ]
            for future in futures:
                df = future.result()
                if df is not None:
                    rows.append(df)

    if not rows:
        return pd.DataFrame(columns=["sample_id", "time", "channel", "value"])

    df = pd.concat(rows, ignore_index=True)
    return df.sort_values(["sample_id", "time", "channel"]).reset_index(drop=True)


def to_long_frame(records: list[dict], channel_names: list[str], n_jobs: int = 1) -> pd.DataFrame:
    """Convert irregular samples into a tidy long dataframe.

    Args:
        records: List of samples, each as a dict containing keys ``t`` and ``x``.
        channel_names: Channel names corresponding to columns of ``x``.
        n_jobs: Number of worker threads. Use values >1 to parallelize per-sample
            conversion on large datasets.
    """
    return _to_long_frame_impl(records, channel_names, n_jobs=n_jobs)


def get_sample_lengths(records: list[dict]) -> pd.DataFrame:
    """Per-sample statistics for irregular records."""
    stats = []
    for i, rec in enumerate(records):
        t = np.asarray(rec["t"], dtype=float)
        x = np.asarray(rec["x"], dtype=float)
        mask = np.isfinite(x)
        n_obs = int(mask.sum())
        n_time = int(len(t))
        duration = float(np.nanmax(t) - np.nanmin(t)) if n_time > 1 else 0.0
        stats.append(
            {
                "sample_id": i,
                "n_timestamps": n_time,
                "n_observations": n_obs,
                "duration": duration,
            }
        )
    return pd.DataFrame(stats)


def compute_dataset_summary(records: list[dict], channel_names: list[str]) -> pd.Series:
    lens = get_sample_lengths(records)
    n_samples = len(records)
    n_channels = len(channel_names)

    if n_samples == 0:
        return pd.Series(
            {
                "n_samples": 0,
                "n_channels": n_channels,
                "mean_timestamps": np.nan,
                "median_timestamps": np.nan,
                "mean_observations": np.nan,
                "median_observations": np.nan,
                "mean_duration": np.nan,
            }
        )

    return pd.Series(
        {
            "n_samples": n_samples,
            "n_channels": n_channels,
            "mean_timestamps": lens["n_timestamps"].mean(),
            "median_timestamps": lens["n_timestamps"].median(),
            "mean_observations": lens["n_observations"].mean(),
            "median_observations": lens["n_observations"].median(),
            "mean_duration": lens["duration"].mean(),
        }
    )


def plot_sample_length_distributions(lens: pd.DataFrame, title_prefix: str = "") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    sns.histplot(lens["n_timestamps"], bins=30, kde=True, ax=axes[0], color="#3A86FF")
    axes[0].set_title(f"{title_prefix}timestamps per sample")

    sns.histplot(lens["n_observations"], bins=30, kde=True, ax=axes[1], color="#2A9D8F")
    axes[1].set_title(f"{title_prefix}observations per sample")

    sns.histplot(lens["duration"], bins=30, kde=True, ax=axes[2], color="#E76F51")
    axes[2].set_title(f"{title_prefix}sample duration")

    plt.tight_layout()


def plot_channel_missingness(long_df: pd.DataFrame, channel_names: list[str], n_samples: int = 200) -> None:
    """Plot per-sample per-channel observation counts heatmap."""
    if long_df.empty:
        print("No data available for missingness heatmap.")
        return

    sampled_ids = sorted(long_df["sample_id"].drop_duplicates().sample(min(n_samples, long_df["sample_id"].nunique()), random_state=0).tolist())
    subset = long_df[long_df["sample_id"].isin(sampled_ids)]

    pivot = (
        subset.groupby(["sample_id", "channel"]).size().unstack(fill_value=0)
        .reindex(columns=channel_names, fill_value=0)
    )

    plt.figure(figsize=(max(10, len(channel_names) * 0.5), 8))
    sns.heatmap(np.log1p(pivot), cmap="YlGnBu", cbar_kws={"label": "log(1 + count)"})
    plt.title("Observation density heatmap (sample x channel)")
    plt.xlabel("Channel")
    plt.ylabel("Sample ID")
    plt.tight_layout()


def plot_interarrival_distribution(records: list[dict], time_unit: str = "time") -> None:
    deltas = []
    for rec in records:
        t = np.asarray(rec["t"], dtype=float)
        if len(t) > 1:
            d = np.diff(np.sort(t))
            d = d[np.isfinite(d)]
            d = d[d >= 0]
            deltas.extend(d.tolist())

    if not deltas:
        print("No inter-arrival statistics available.")
        return

    plt.figure(figsize=(8, 4))
    sns.histplot(deltas, bins=60, kde=True, color="#F4A261")
    plt.title("Inter-arrival time distribution")
    plt.xlabel(f"Gap ({time_unit})")
    plt.ylabel("Count")
    plt.tight_layout()


def plot_multichannel_trajectories(
    long_df: pd.DataFrame,
    channels: list[str],
    n_samples: int = 8,
    max_points_per_sample: int = 600,
    col_wrap: int = 2,
    height: float = 4.0,
    aspect: float = 1.8,
    show_legend: bool = True,
    plot_kind: str = "line",
    normalize_time_per_sample: bool = False,
) -> None:
    required_cols = {"sample_id", "time", "channel", "value"}
    missing_cols = sorted(required_cols.difference(long_df.columns))
    if missing_cols:
        print(f"Cannot plot trajectories. Missing columns: {missing_cols}")
        print(f"Available columns: {list(long_df.columns)}")
        return

    if long_df.empty:
        print("No points to plot.")
        return

    available_channels = set(long_df["channel"].dropna().astype(str).unique().tolist())
    requested_channels = [str(ch) for ch in channels]
    selected_channels = [ch for ch in requested_channels if ch in available_channels]
    if not selected_channels:
        fallback_channels = (
            long_df["channel"].astype(str).value_counts().head(4).index.tolist()
        )
        print(
            "Requested channels are not present in long_df. "
            f"Falling back to top channels: {fallback_channels}"
        )
        selected_channels = fallback_channels

    ids = long_df["sample_id"].drop_duplicates().sample(min(n_samples, long_df["sample_id"].nunique()), random_state=0).tolist()
    plot_df = long_df[(long_df["sample_id"].isin(ids)) & (long_df["channel"].astype(str).isin(selected_channels))].copy()

    if plot_df.empty:
        print("No matching points after filtering by sample/channel.")
        print(f"Selected channels: {selected_channels}")
        return

    if len(plot_df) > max_points_per_sample * len(ids) * max(1, len(selected_channels)):
        sampled_parts: list[pd.DataFrame] = []
        # Explicitly sample each (sample_id, channel) group to keep columns stable
        # across pandas versions where groupby.apply may drop grouping columns.
        for sid in ids:
            for ch in selected_channels:
                sub = plot_df[
                    (plot_df["sample_id"] == sid)
                    & (plot_df["channel"].astype(str) == ch)
                ]
                if sub.empty:
                    continue
                if len(sub) > max_points_per_sample:
                    sub = sub.sample(n=max_points_per_sample, random_state=0)
                sampled_parts.append(sub)

        if sampled_parts:
            plot_df = pd.concat(sampled_parts, ignore_index=True)

    # Use matplotlib directly to avoid seaborn relplot variable parsing issues
    # across different seaborn versions/environments.
    plot_df = plot_df.loc[:, ["sample_id", "time", "channel", "value"]].copy()
    plot_df["sample_id"] = plot_df["sample_id"].astype(str)
    plot_df["channel"] = plot_df["channel"].astype(str)

    if normalize_time_per_sample:
        tmin = plot_df.groupby("sample_id")["time"].transform("min")
        tmax = plot_df.groupby("sample_id")["time"].transform("max")
        denom = (tmax - tmin).replace(0, np.nan)
        plot_df["time_plot"] = ((plot_df["time"] - tmin) / denom).fillna(0.0)
        x_col = "time_plot"
        x_label = "normalized time (0-1)"
    else:
        x_col = "time"
        x_label = "time"

    channels_order = [ch for ch in selected_channels if ch in set(plot_df["channel"])]
    if not channels_order:
        print("No channels left to plot after preprocessing.")
        return

    n_cols = max(1, col_wrap)
    n_rows = int(np.ceil(len(channels_order) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * height * aspect, n_rows * height),
        sharex=True,
        squeeze=False,
    )

    sample_ids = sorted(plot_df["sample_id"].unique().tolist())
    palette = sns.color_palette("tab10", n_colors=max(3, len(sample_ids)))
    color_map = {sid: palette[i % len(palette)] for i, sid in enumerate(sample_ids)}

    for idx, ch in enumerate(channels_order):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        sub = plot_df[plot_df["channel"] == ch].sort_values(["sample_id", "time"])
        for sid, grp in sub.groupby("sample_id"):
            if plot_kind == "scatter":
                ax.scatter(
                    grp[x_col].to_numpy(),
                    grp["value"].to_numpy(),
                    color=color_map[sid],
                    alpha=0.55,
                    s=9,
                    linewidths=0,
                )
            else:
                ax.plot(
                    grp[x_col].to_numpy(),
                    grp["value"].to_numpy(),
                    color=color_map[sid],
                    alpha=0.8,
                    linewidth=1.1,
                )
        ax.set_title(ch)
        ax.set_xlabel(x_label)
        ax.set_ylabel("value")
        ax.grid(alpha=0.25)

    # Hide any unused subplot slots.
    for idx in range(len(channels_order), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")

    if show_legend and sample_ids:
        from matplotlib.lines import Line2D

        handles = [
            Line2D([0], [0], color=color_map[sid], lw=2, label=f"sample {sid}")
            for sid in sample_ids
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=min(4, len(handles)),
            frameon=True,
            title="Trajectories",
            bbox_to_anchor=(0.5, 1.08),
        )

    fig.suptitle("Multi-sample trajectories by channel", y=1.13)
    fig.tight_layout()


def plot_channel_correlation(long_df: pd.DataFrame, channel_names: list[str], max_samples: int = 300) -> None:
    if long_df.empty:
        print("No data for correlation plot.")
        return

    sampled_ids = long_df["sample_id"].drop_duplicates()
    if len(sampled_ids) > max_samples:
        sampled_ids = sampled_ids.sample(max_samples, random_state=0)
    sampled_ids = set(sampled_ids.tolist())

    pivot = (
        long_df[long_df["sample_id"].isin(sampled_ids)]
        .groupby(["sample_id", "channel"])["value"].mean()
        .unstack()
        .reindex(columns=channel_names)
    )

    corr = pivot.corr(min_periods=max(5, int(0.1 * len(pivot))))

    plt.figure(figsize=(max(8, 0.45 * len(channel_names)), max(6, 0.45 * len(channel_names))))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title("Channel correlation (sample-level mean)")
    plt.tight_layout()


def records_from_task_dataset(task_dataset: Iterable) -> list[dict]:
    """Convert APN task datasets into a unified {t, x} format."""
    records = []
    for sample in task_dataset:
        t, x, _t_target = sample.inputs
        y = sample.targets
        t_np = np.concatenate([np.asarray(t, dtype=float), np.asarray(_t_target, dtype=float)])
        x_np = np.concatenate([np.asarray(x, dtype=float), np.asarray(y, dtype=float)], axis=0)
        order = np.argsort(t_np)
        records.append({"t": t_np[order], "x": x_np[order]})
    return records


def records_from_human_activity(human_activity_dataset) -> list[dict]:
    records = []
    for _rid, tt, vals, _mask in human_activity_dataset:
        records.append({"t": np.asarray(tt, dtype=float), "x": np.asarray(vals, dtype=float)})
    return records
