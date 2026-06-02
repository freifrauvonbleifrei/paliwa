// SPDX-FileCopyrightText: 2026 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include <benchmark/benchmark.h>

#include <ddc/ddc.hpp>
#include <Kokkos_Core.hpp>

#include "../paliwa/paliwa_dimensions.hpp"

#include <array>
#include <numeric>
#include <string>

namespace paliwa_benchmarks {

using BDim0 = paliwa::DDimA;
using BDim1 = paliwa::DDimB;
using BDim2 = paliwa::DDimC;
using BDim3 = paliwa::DDimD;
using BDim4 = paliwa::DDimE;
using BDim5 = paliwa::DDimF;
using BDim6 = paliwa::DDimG;
using BDim7 = paliwa::DDimH;
using BDim8 = paliwa::DDimI;
using BDim9 = paliwa::DDimJ;
using BDimX = paliwa::DDimX;
using BDimY = paliwa::DDimY;
using BDimZ = paliwa::DDimZ;

template <class SDDom>
static void set_memory_bandwidth_counter(
        benchmark::State &state, SDDom const &sd, long long passes = 1) {
    long long bytes = static_cast<long long>(state.iterations()) *
                      static_cast<long long>(sd.size()) *
                      static_cast<long long>(sizeof(double)) *
                      passes;
    state.counters["MemBW"] = benchmark::Counter(static_cast<double>(bytes), benchmark::Counter::kIsRate);
}

template <class SDDom>
static void set_working_set_counter(benchmark::State &state, SDDom const &sd) {
    long long bytes = static_cast<long long>(sd.size()) * static_cast<long long>(sizeof(double));
    state.counters["WorkingSet"] = benchmark::Counter(static_cast<double>(bytes));
}

template <class... DDims>
static ddc::StridedDiscreteDomain<DDims...> make_strided_domain_from_extent(long int extent_xy) {
    using SDDom = ddc::StridedDiscreteDomain<DDims...>;
    using DVect = ddc::DiscreteVector<DDims...>;

    DVect const extents(std::array<long int, sizeof...(DDims)>({((void)DDims{}, extent_xy)...}));
    DVect const strides(std::array<long int, sizeof...(DDims)>({((void)DDims{}, 1L)...}));

    return SDDom(ddc::DiscreteElement<DDims...>(), extents, strides);
}

template <class ChunkView, class SDDom>
static void fill_reference_data(ChunkView &view, SDDom const &sd) {
    ddc::host_for_each(sd, [&](auto const idx) { view(idx) = 1.0; });
}

} // namespace paliwa_benchmarks
