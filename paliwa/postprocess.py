#!/usr/bin/env/python3

# SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import matplotlib.pyplot as plt
from icecream import ic


def filename_from_level(level):
    level_str = "_".join([str(l) for l in level]) + "_" + str(len(level)) + "d"
    return "strided_grid_" + level_str + ".raw"


if __name__ == "__main__":
    # combination_scheme = [[4,6,7], [5,5,7], [5,6,6], [4,5,6]]
    # combination_scheme = [[2, 5], [3, 4], [4, 3], [2, 4], [3, 3]]
    combination_scheme = [[2, 11], [2, 10], [3, 10], [3, 9], [4, 9], [4, 8], [5, 8], [5, 7], 
        [6, 7], [6, 6], [7, 6], [7, 5], [8, 5], [8, 4], [9, 4], [9, 3], [10, 3]]
    minimum_level = np.min(combination_scheme, axis=0)
    maximum_level = np.max(combination_scheme, axis=0)
    ic(minimum_level, maximum_level)
    extent = [0.0, 1.0, 0.0, 1.0]
    nrows = maximum_level[0] - minimum_level[0] + 1
    ncols = maximum_level[1] - minimum_level[1] + 1
    ic(nrows, ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    for level in combination_scheme:
        filename = filename_from_level(level)
        ic(filename)
        data = np.fromfile(filename, dtype=np.float64)
        shape = tuple([2**l for l in level])
        data = data.reshape(shape)
        ic(data.shape)
        # plot slices
        ax = plt.subplot2grid(
            (nrows, ncols),
            (level[0] - minimum_level[0], level[1] - minimum_level[1]),
            fig=fig,
        )
        ax.imshow(data, aspect="auto", origin="lower",
            vmin=-1.0,
            vmax=1.0,)
        ax.set_title(f"Level: {level}")
    # print maximum level
    full_grid_filename =  filename_from_level(maximum_level)
    full_grid_filename = full_grid_filename.replace("strided_grid", "full_grid")
    ic(full_grid_filename)
    full_grid_data = np.fromfile(full_grid_filename, dtype=np.float64)
    full_grid_shape = tuple([2**l for l in maximum_level])
    full_grid_data = full_grid_data.reshape(full_grid_shape)
    ic(full_grid_data.shape)
    ax = plt.subplot2grid(
        (nrows, ncols),
        (nrows -1, ncols -1),
        fig=fig,
    )
    ax.imshow(full_grid_data, aspect="auto", origin="lower",
        vmin=-1.0,
        vmax=1.0,)
    ax.set_title(f"Combined: {maximum_level}")
    # plt.show()
    plt.savefig("combination_scheme.png")
