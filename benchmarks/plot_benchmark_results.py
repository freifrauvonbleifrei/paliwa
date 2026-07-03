#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: LGPL-3.0-or-later


"""Plot Google Benchmark JSON results.

Usage:
    python3 benchmarks/plot_benchmark_results.py results.json --output-dir plots

The script groups entries by benchmark family, uses the first benchmark argument
as the x-axis, and draws one series per second argument when present.
It plots any selected metrics that are available in the input file.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib
except ImportError as exc:  # pragma: no cover - user-facing runtime guard
    raise SystemExit(
        "matplotlib is required. Install it with: python3 -m pip install -r benchmarks/requirements.txt"
    ) from exc

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_METRICS = ["real_time", "MemBW", "ComBW"]


DISPLAY_NAMES = {
    "real_time": "real time [ns]",
    "MemBW": "memory bandwidth [bytes/s]",
    "WorkingSet": "working set size [bytes]",
    "ComBW": "communication bandwidth [bytes/s]",
}

X_AXIS_LABEL = "working set size [bytes]"


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def parse_benchmark_name(name: str) -> tuple[str, list[str]]:
    parts = name.split("/")
    return parts[0], parts[1:]


def maybe_number(value):
    try:
        if isinstance(value, str) and value.strip() == "":
            return value
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return int(number)
        return number
    except (TypeError, ValueError):
        return value


def extract_metric(entry: dict, metric: str):
    if metric in entry:
        return entry[metric]
    counters = entry.get("counters", {})
    if metric not in counters:
        return None
    value = counters[metric]
    if isinstance(value, dict):
        return value.get("value")
    return value


def load_entries(paths: list[Path]) -> list[dict]:
    entries = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        entries.extend(payload.get("benchmarks", []))
    return entries


def group_entries(entries: list[dict]) -> dict[str, dict[str, list[dict]]]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        family, args = parse_benchmark_name(entry.get("name", ""))
        if not family:
            continue
        series = args[1] if len(args) > 1 else "default"
        grouped[family][series].append(entry)
    return grouped


def plot_family(family: str, series_map: dict[str, list[dict]], metrics: list[str], output_dir: Path):
    available_metrics = [
        metric
        for metric in metrics
        if any(extract_metric(entry, metric) is not None for entries in series_map.values() for entry in entries)
    ]
    if not available_metrics:
        return

    fig, axes = plt.subplots(len(available_metrics), 1, figsize=(10, 4 * len(available_metrics)), squeeze=False)
    fig.suptitle(family)

    for axis, metric in zip(axes.flat, available_metrics):
        for series_name, entries in sorted(series_map.items()):
            points = []
            for entry in entries:
                args = parse_benchmark_name(entry.get("name", ""))[1]
                if not args:
                    continue
                x_value = maybe_number(extract_metric(entry, "WorkingSet"))
                y_value = maybe_number(extract_metric(entry, metric))
                if x_value is None or y_value is None:
                    continue
                points.append((x_value, y_value))

            if not points:
                continue

            points.sort(key=lambda item: item[0])
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            axis.plot(xs, ys, marker="o", label=series_name)

        axis.set_title(DISPLAY_NAMES.get(metric, metric))
        axis.set_xlabel(X_AXIS_LABEL)
        axis.set_ylabel(DISPLAY_NAMES.get(metric, metric))
        axis.set_xscale("log", base=2)
        axis.grid(True, alpha=0.3)
        if len(series_map) > 1:
            axis.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{sanitize_name(family)}.png"
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    print(output_file)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Google Benchmark JSON files")
    parser.add_argument("--output-dir", type=Path, default=Path("plots"), help="Directory for generated plots")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to plot if available",
    )
    args = parser.parse_args()

    entries = load_entries(args.inputs)
    grouped = group_entries(entries)

    for family, series_map in sorted(grouped.items()):
        plot_family(family, series_map, args.metrics, args.output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())