// SPDX-FileCopyrightText: 2026 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "benchmark_common.hpp"

namespace paliwa_benchmarks {

template <class... DDims>
static void BM_CreateMirrorAndCopy(benchmark::State &state) {
        using SDDom = ddc::StridedDiscreteDomain<DDims...>;

        long int const extent_xy = state.range(0);

                SDDom const strided_domain = make_strided_domain_from_extent<DDims...>(extent_xy);

    ddc::Chunk baseline_chunk("baseline_copy", strided_domain, ddc::HostAllocator<double>());
    auto baseline = baseline_chunk.span_view();
    fill_reference_data(baseline, strided_domain);

    for (auto _ : state) {
        auto copied = ddc::create_mirror_and_copy(Kokkos::SharedHostPinnedSpace(), baseline);
        benchmark::DoNotOptimize(copied);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<long long>(strided_domain.size()));
    set_memory_bandwidth_counter(state, strided_domain);
        set_working_set_counter(state, strided_domain);
}

template <class... DDims>
static void BM_CopyOnly(benchmark::State &state) {
                using SDDom = ddc::StridedDiscreteDomain<DDims...>;

        long int const extent_xy = state.range(0);

        SDDom const strided_domain = make_strided_domain_from_extent<DDims...>(extent_xy);

    ddc::Chunk baseline_chunk("baseline_bandwidth", strided_domain, ddc::HostAllocator<double>());
    auto baseline = baseline_chunk.span_view();

    fill_reference_data(baseline, strided_domain);

        auto working_chunk = ddc::create_mirror(Kokkos::SharedHostPinnedSpace(), baseline);

    for (auto _ : state) {
                ddc::parallel_deepcopy(working_chunk, baseline);
                benchmark::DoNotOptimize(working_chunk);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<long long>(strided_domain.size()));
    set_memory_bandwidth_counter(state, strided_domain);
        set_working_set_counter(state, strided_domain);
}

static void BM_CreateMirrorAndCopy2D(benchmark::State &state) {
        BM_CreateMirrorAndCopy<BDimX, BDimY>(state);
}

static void BM_CreateMirrorAndCopy4D(benchmark::State &state) {
        BM_CreateMirrorAndCopy<BDim0, BDim1, BDim2, BDim3>(state);
}

static void BM_CreateMirrorAndCopy6D(benchmark::State &state) {
        BM_CreateMirrorAndCopy<BDim0, BDim1, BDim2, BDim3, BDim4, BDim5>(state);
}

static void BM_CopyOnly2D(benchmark::State &state) {
        BM_CopyOnly<BDimX, BDimY>(state);
}

static void BM_CopyOnly4D(benchmark::State &state) {
        BM_CopyOnly<BDim0, BDim1, BDim2, BDim3>(state);
}

static void BM_CopyOnly6D(benchmark::State &state) {
        BM_CopyOnly<BDim0, BDim1, BDim2, BDim3, BDim4, BDim5>(state);
}

} // namespace paliwa_benchmarks

BENCHMARK(paliwa_benchmarks::BM_CreateMirrorAndCopy2D)
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

BENCHMARK(paliwa_benchmarks::BM_CreateMirrorAndCopy4D)
                ->Arg(4)
                ->Arg(8)
                ->Arg(16)
                ->Arg(32)
                ->Arg(64)
                ->Arg(128);

BENCHMARK(paliwa_benchmarks::BM_CreateMirrorAndCopy6D)
                ->Arg(4)
                ->Arg(6)
                ->Arg(8)
                ->Arg(12)
                ->Arg(16)
                ->Arg(24)
                ->Arg(32);

BENCHMARK(paliwa_benchmarks::BM_CopyOnly2D)
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

BENCHMARK(paliwa_benchmarks::BM_CopyOnly4D)
                ->Arg(4)
                ->Arg(8)
                ->Arg(16)
                ->Arg(32)
                ->Arg(64)
                ->Arg(128);

BENCHMARK(paliwa_benchmarks::BM_CopyOnly6D)
                ->Arg(4)
                ->Arg(8)
                ->Arg(12)
                ->Arg(16)
                ->Arg(24)
                ->Arg(32);

