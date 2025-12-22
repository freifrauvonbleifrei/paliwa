// SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once

#include <vector>

namespace paliwa {
// TODO consider making this constexpr frozen::map ?
static const std::map<std::string,
                      std::vector<std::pair<int, std::array<double, 3>>>>
    lifting_wavelet_filter_offsets_and_coefficients = {
        {"hat", {{1, {-0.5, 1.0, -0.5}}}},
        {"biorthogonal", {{1, {-0.5, 1.0, -0.5}}, {0, {0.25, 1.0, 0.25}}}},
        {"fullweighting", {{0, {0.25, 0.5, 0.25}}, {1, {-0.5, 1.0, -0.5}}}},
};
static const std::map<std::string,
                      std::vector<std::pair<int, std::array<double, 3>>>>
    lifting_wavelet_reconstruct_offsets_and_coefficients = {
        {"hat", {{1, {0.5, 1.0, 0.5}}}},
        {"biorthogonal", {{0, {-0.25, 1.0, 0.25}}, {1, {0.5, 1.0, 0.5}}}},
        {"fullweighting", {{1, {0.5, 1.0, 0.5}}, {0, {-0.5, 2.0, -0.5}}}},
};
} // namespace paliwa