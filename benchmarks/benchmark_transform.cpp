// SPDX-FileCopyrightText: 2026 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "benchmark_common.hpp"

#include "../paliwa/paliwa_domains.hpp"
#include "../paliwa/paliwa_transform.hpp"

#ifdef PALIWA_WITH_LIKWID
#include <likwid-marker.h>
#endif

namespace paliwa_benchmarks {

template <class... DDims>
static void BM_HierarchizeDehierarchizeRoundtrip(benchmark::State &state) {
    using SDDom = ddc::StridedDiscreteDomain<DDims...>;
    using DElem = ddc::DiscreteElement<DDims...>;
    using DVect = ddc::DiscreteVector<DDims...>;

        long int const extent_xy = state.range(0);
        long int level_val = 0;
        for (long int reduced_extent = extent_xy; reduced_extent > 1; reduced_extent >>= 1) {
                ++level_val;
        }

    DVect const level(std::array<long int, sizeof...(DDims)>{((void)DDims{}, level_val)...});
    DVect const minimum_level(std::array<long int, sizeof...(DDims)>{((void)DDims{}, 1L)...});
    DVect const maximum_level(
            std::array<long int, sizeof...(DDims)>{((void)DDims{}, level_val + 1)...});

        SDDom const strided_domain = make_strided_domain_from_extent<DDims...>(extent_xy);

    ddc::Chunk baseline_chunk("baseline_roundtrip", strided_domain, ddc::HostAllocator<double>());
    ddc::Chunk working_chunk("working_roundtrip", strided_domain, ddc::HostAllocator<double>());
    auto baseline = baseline_chunk.span_view();
    auto working = working_chunk.span_view();

    fill_reference_data(baseline, strided_domain);

#ifdef PALIWA_WITH_LIKWID
    LIKWID_MARKER_INIT;
    LIKWID_MARKER_REGISTER("transform_roundtrip");
#endif

    for (auto _ : state) {
        state.PauseTiming();
        ddc::host_for_each(strided_domain, [&](DElem const idx) { working(idx) = baseline(idx); });
        state.ResumeTiming();

#ifdef PALIWA_WITH_LIKWID
        LIKWID_MARKER_START("transform_roundtrip");
#endif

        paliwa::hierarchize(
                working, level, minimum_level, maximum_level,
                "biorthogonal", Kokkos::DefaultExecutionSpace());
        paliwa::dehierarchize(
                working, level, minimum_level, maximum_level,
                "biorthogonal", Kokkos::DefaultExecutionSpace());

#ifdef PALIWA_WITH_LIKWID
        LIKWID_MARKER_STOP("transform_roundtrip");
#endif

        benchmark::DoNotOptimize(working);
        benchmark::ClobberMemory();
    }

#ifdef PALIWA_WITH_LIKWID
    LIKWID_MARKER_CLOSE;
#endif

    state.SetItemsProcessed(
            state.iterations() * static_cast<long long>(strided_domain.size()) * 2LL);
    set_memory_bandwidth_counter(state, strided_domain, 2LL);
        set_working_set_counter(state, strided_domain);
}

static void BM_HierarchizeDehierarchizeRoundtrip2D(benchmark::State &state) {
    BM_HierarchizeDehierarchizeRoundtrip<BDimX, BDimY>(state);
}

static void BM_HierarchizeDehierarchizeRoundtrip3D(benchmark::State &state) {
    BM_HierarchizeDehierarchizeRoundtrip<BDimX, BDimY, BDimZ>(state);
}

static void BM_HierarchizeDehierarchizeRoundtrip4D(benchmark::State &state) {
    BM_HierarchizeDehierarchizeRoundtrip<BDim0, BDim1, BDim2, BDimX>(state);
}

} // namespace paliwa_benchmarks

BENCHMARK(paliwa_benchmarks::BM_HierarchizeDehierarchizeRoundtrip2D)
        ->Arg(8)
        ->Arg(16)
        ->Arg(32)
        ->Arg(64)
        ->Arg(128)
        ->Arg(256)
        ->Arg(512)
        ->Arg(1024)
        ->Arg(2048)
        ->Arg(4096)
        ->Arg(8192)
        ->Arg(16384);

BENCHMARK(paliwa_benchmarks::BM_HierarchizeDehierarchizeRoundtrip3D)
        ->Arg(8)
        ->Arg(16)
        ->Arg(32)
        ->Arg(64)
        ->Arg(128)
        ->Arg(256)
        ->Arg(512)
        ->Arg(1024);

BENCHMARK(paliwa_benchmarks::BM_HierarchizeDehierarchizeRoundtrip4D)
        ->Arg(8)
        ->Arg(16)
        ->Arg(32)
        ->Arg(64)
        ->Arg(128);

