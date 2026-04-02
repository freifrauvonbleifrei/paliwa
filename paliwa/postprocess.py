#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Visualize combination technique component grids.

Supports both serial (single-rank) and distributed (multi-rank) raw files.

Serial files:      strided_grid_{level}_{dim}d.raw
Distributed files: distributed_grid_{level}_{dim}d_rank{r}.raw

Usage:
    python postprocess.py                         # serial (default)
    python postprocess.py --distributed 2 2       # distributed with 2x2 ranks
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


def level_str(level):
    return "_".join(str(l) for l in level) + f"_{len(level)}d"


def load_serial_grid(level):
    filename = f"strided_grid_{level_str(level)}.raw"
    if not os.path.exists(filename):
        return None
    data = np.fromfile(filename, dtype=np.float64)
    return data.reshape(tuple(2**l for l in level))


def load_distributed_grid(level, maximum_level, n_ranks_per_dim,
                          prefix="distributed_grid"):
    """Reassemble a distributed grid from per-rank raw files."""
    global_points = [2**l for l in level]
    local_points = [gp // nr for gp, nr in zip(global_points, n_ranks_per_dim)]
    n_ranks = 1
    for nr in n_ranks_per_dim:
        n_ranks *= nr

    full = np.zeros(global_points, dtype=np.float64)
    for rank in range(n_ranks):
        filename = f"{prefix}_{level_str(level)}_rank{rank}.raw"
        if not os.path.exists(filename):
            return None
        data = np.fromfile(filename, dtype=np.float64).reshape(tuple(local_points))

        # Cartesian coordinates (row-major rank ordering)
        coords = []
        r = rank
        for d in range(len(level) - 1, -1, -1):
            coords.insert(0, r % n_ranks_per_dim[d])
            r //= n_ranks_per_dim[d]

        slices = tuple(
            slice(c * lp, (c + 1) * lp) for c, lp in zip(coords, local_points)
        )
        full[slices] = data

    return full


def load_grid(level, maximum_level, n_ranks_per_dim):
    if n_ranks_per_dim is None:
        return load_serial_grid(level)
    return load_distributed_grid(level, maximum_level, n_ranks_per_dim)


def plot_combination_scheme(combination_scheme, maximum_level, n_ranks_per_dim,
                            output_filename):
    minimum_level = np.min(combination_scheme, axis=0)
    nrows = maximum_level[0] - minimum_level[0] + 1
    ncols = maximum_level[1] - minimum_level[1] + 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 10))
    if n_ranks_per_dim is None:
        fig.suptitle("Combination Technique (serial)", fontsize=14)
    else:
        fig.suptitle(
            f"Combination Technique (distributed, "
            f"{'x'.join(str(n) for n in n_ranks_per_dim)} ranks)",
            fontsize=14,
        )

    # Turn off all axes first
    for ax_row in (axes if axes.ndim == 2 else [axes]):
        for ax in (ax_row if hasattr(ax_row, '__iter__') else [ax_row]):
            ax.axis("off")

    for level in combination_scheme:
        data = load_grid(level, maximum_level, n_ranks_per_dim)
        if data is None:
            print(f"  Warning: missing data for level {level}")
            continue
        r = level[0] - minimum_level[0]
        c = level[1] - minimum_level[1]
        ax = axes[r, c] if axes.ndim == 2 else axes[max(r, c)]
        ax.imshow(data, aspect="auto", origin="lower", vmin=-1.0, vmax=1.0)
        ax.set_title(f"{level}", fontsize=8)
        ax.axis("on")

    # Load and plot the combined full grid
    if n_ranks_per_dim is None:
        full_filename = f"full_grid_{level_str(maximum_level)}.raw"
        if os.path.exists(full_filename):
            full_data = np.fromfile(full_filename, dtype=np.float64)
            full_data = full_data.reshape(tuple(2**l for l in maximum_level))
            ax = axes[nrows - 1, ncols - 1]
            ax.imshow(full_data, aspect="auto", origin="lower", vmin=-1.0, vmax=1.0)
            ax.set_title(f"Combined {list(maximum_level)}", fontsize=8)
            ax.axis("on")
    else:
        full_data = load_distributed_grid(
            maximum_level, maximum_level, n_ranks_per_dim,
            prefix="distributed_full_grid")
        if full_data is not None:
            ax = axes[nrows - 1, ncols - 1]
            ax.imshow(full_data, aspect="auto", origin="lower", vmin=-1.0, vmax=1.0)
            ax.set_title(f"Combined {list(maximum_level)}", fontsize=8)
            ax.axis("on")

    fig.tight_layout()
    fig.savefig(output_filename, dpi=150)
    print(f"Wrote {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize combination technique component grids"
    )
    parser.add_argument(
        "--distributed", nargs="+", type=int, default=None,
        help="Number of MPI ranks per dimension (e.g. --distributed 2 2)",
    )
    args = parser.parse_args()

    if args.distributed is not None:
        # Distributed test: max_level={6,7}, lmin={2,3}
        combination_scheme = [
            [2, 7], [3, 6], [4, 5], [5, 4], [6, 3],
            [2, 6], [3, 5], [4, 4], [5, 3],
        ]
        maximum_level = [6, 7]
        plot_combination_scheme(
            combination_scheme, maximum_level, args.distributed,
            "combination_scheme_distributed.png",
        )
    else:
        # Serial test: max_level={10,11}, lmin={2,3}
        combination_scheme = [
            [2, 11], [3, 10], [4, 9], [5, 8], [6, 7],
            [7, 6], [8, 5], [9, 4], [10, 3],
            [2, 10], [3, 9], [4, 8], [5, 7], [6, 6], [7, 5], [8, 4], [9, 3],
        ]
        maximum_level = [10, 11]
        plot_combination_scheme(
            combination_scheme, maximum_level, None,
            "combination_scheme.png",
        )
