// SPDX-FileCopyrightText: 2026 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <algorithm>
#include <array>

#include <benchmark/benchmark.h>

#include <ddc/ddc.hpp>

#include "../paliwa/paliwa_domains.hpp"
#include "../paliwa/paliwa_transform.hpp"

struct BX {
};
struct BY {
};
struct BDimX : ddc::UniformPointSampling<BX> {
};
struct BDimY : ddc::UniformPointSampling<BY> {
};

struct BZ {};
struct BDimZ : ddc::UniformPointSampling<BZ> {};
struct B0 {};
struct BDim0 : ddc::UniformPointSampling<B0> {};
struct B1 {};
struct BDim1 : ddc::UniformPointSampling<B1> {};
struct B2 {};
struct BDim2 : ddc::UniformPointSampling<B2> {};

namespace {

template <class SpanType, class DomainType>
void fill_reference_data(SpanType span, DomainType const& domain)
{
        using DElem = typename DomainType::discrete_element_type;
        ddc::host_for_each(domain, [&](DElem const ixy) {
                // Use the geometric coordinate in each dimension to form a value
                auto coord = ddc::coordinate(ixy).array();
                double val = 1.0;
                for (double c : coord) {
                        val *= (1.0 + 0.1 * c);
                }
                span(ixy) = val;
        });
}

static void BM_RestrictStridedWithDiscrete2D(benchmark::State& state)
{
    using SDDom = ddc::StridedDiscreteDomain<BDimX, BDimY>;
    using DDom = ddc::DiscreteDomain<BDimX, BDimY>;

    long int const level_x = state.range(0);
    long int const level_y = state.range(1);
    std::array<long int, 2> const level = {level_x, level_y};
    std::array<long int, 2> const maximum_level = {level_x + 1, level_y + 1};

    SDDom const strided_domain =
            paliwa::strided_domain_from_level<BDimX, BDimY>(level, maximum_level);

    long int const nx = 1L << level_x;
    long int const ny = 1L << level_y;
    DDom const local_domain(
            ddc::DiscreteDomain<BDimX>(
                    ddc::DiscreteElement<BDimX>(nx / 4),
                    ddc::DiscreteVector<BDimX>(std::max(2L, nx / 2))),
            ddc::DiscreteDomain<BDimY>(
                    ddc::DiscreteElement<BDimY>(ny / 4),
                    ddc::DiscreteVector<BDimY>(std::max(2L, ny / 2))));

    for (auto _ : state) {
                auto restricted =
                paliwa::restrict_strided_with_discrete(strided_domain, local_domain);
        benchmark::DoNotOptimize(restricted);
    }

    state.SetItemsProcessed(state.iterations());
}

static void BM_HierarchizeDehierarchizeRoundtrip2D(benchmark::State& state)
{
    using SDDom = ddc::StridedDiscreteDomain<BDimX, BDimY>;
    using DElem = ddc::DiscreteElement<BDimX, BDimY>;
    using DVect = ddc::DiscreteVector<BDimX, BDimY>;

    long int const level_xy = state.range(0);

    DVect const level(std::array<long int, 2>({level_xy, level_xy}));
    DVect const minimum_level(std::array<long int, 2>({1, 1}));
    DVect const maximum_level(std::array<long int, 2>({level_xy + 1, level_xy + 1}));

    SDDom const strided_domain = paliwa::strided_domain_from_level<BDimX, BDimY>(
            ddc::detail::array(level),
            ddc::detail::array(maximum_level));

    ddc::Chunk baseline_chunk("baseline", strided_domain, ddc::HostAllocator<double>());
    ddc::Chunk working_chunk("working", strided_domain, ddc::HostAllocator<double>());
    auto baseline = baseline_chunk.span_view();
    auto working = working_chunk.span_view();

    fill_reference_data(baseline, strided_domain);

    for (auto _ : state) {
        state.PauseTiming();
        ddc::host_for_each(strided_domain, [&](DElem const ixy) { working(ixy) = baseline(ixy); });
        state.ResumeTiming();

        paliwa::hierarchize(
                working,
                strided_domain,
                level,
                minimum_level,
                maximum_level,
                "biorthogonal",
                Kokkos::DefaultHostExecutionSpace());

        paliwa::dehierarchize(
                working,
                strided_domain,
                level,
                minimum_level,
                maximum_level,
                "biorthogonal",
                Kokkos::DefaultHostExecutionSpace());

        benchmark::DoNotOptimize(working);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * strided_domain.size());
}

} // namespace

// ---------- Higher-dimensional benchmarks (3D..6D) ----------

static void BM_HierarchizeDehierarchizeRoundtrip3D(benchmark::State& state)
{
        using SDDom = ddc::StridedDiscreteDomain<BDimX, BDimY, BDimZ>;
        using DElem = ddc::DiscreteElement<BDimX, BDimY, BDimZ>;
        using DVect = ddc::DiscreteVector<BDimX, BDimY, BDimZ>;

        long int const level_val = state.range(0);
        DVect const level(std::array<long int,3>{level_val, level_val, level_val});
        DVect const minimum_level(std::array<long int,3>{1,1,1});
        DVect const maximum_level(std::array<long int,3>{level_val+1, level_val+1, level_val+1});

        SDDom const strided_domain = paliwa::strided_domain_from_level<BDimX,BDimY,BDimZ>(
                        ddc::detail::array(level), ddc::detail::array(maximum_level));

        ddc::Chunk baseline_chunk("baseline3d", strided_domain, ddc::HostAllocator<double>());
        ddc::Chunk working_chunk("working3d", strided_domain, ddc::HostAllocator<double>());
        auto baseline = baseline_chunk.span_view();
        auto working = working_chunk.span_view();

        fill_reference_data(baseline, strided_domain);

        for (auto _ : state) {
                state.PauseTiming();
                ddc::host_for_each(strided_domain, [&](DElem const ixy) { working(ixy) = baseline(ixy); });
                state.ResumeTiming();

                paliwa::hierarchize(working, strided_domain, level, minimum_level, maximum_level,
                                                        "biorthogonal", Kokkos::DefaultHostExecutionSpace());
                paliwa::dehierarchize(working, strided_domain, level, minimum_level, maximum_level,
                                                          "biorthogonal", Kokkos::DefaultHostExecutionSpace());

                benchmark::DoNotOptimize(working);
                benchmark::ClobberMemory();
        }

        state.SetItemsProcessed(state.iterations() * (long long)strided_domain.size());
}

static void BM_HierarchizeDehierarchizeRoundtrip4D(benchmark::State& state)
{
        using SDDom = ddc::StridedDiscreteDomain<BDim0, BDim1, BDim2, BDimX>;
        using DElem = ddc::DiscreteElement<BDim0, BDim1, BDim2, BDimX>;
        using DVect = ddc::DiscreteVector<BDim0, BDim1, BDim2, BDimX>;

        long int const level_val = state.range(0);
        DVect const level(std::array<long int,4>{level_val, level_val, level_val, level_val});
        DVect const minimum_level(std::array<long int,4>{1,1,1,1});
        DVect const maximum_level(std::array<long int,4>{level_val+1, level_val+1, level_val+1, level_val+1});

        SDDom const strided_domain = paliwa::strided_domain_from_level<BDim0,BDim1,BDim2,BDimX>(
                        ddc::detail::array(level), ddc::detail::array(maximum_level));

        ddc::Chunk baseline_chunk("baseline4d", strided_domain, ddc::HostAllocator<double>());
        ddc::Chunk working_chunk("working4d", strided_domain, ddc::HostAllocator<double>());
        auto baseline = baseline_chunk.span_view();
        auto working = working_chunk.span_view();

        fill_reference_data(baseline, strided_domain);

        for (auto _ : state) {
                state.PauseTiming();
                ddc::host_for_each(strided_domain, [&](DElem const ixy) { working(ixy) = baseline(ixy); });
                state.ResumeTiming();

                paliwa::hierarchize(working, strided_domain, level, minimum_level, maximum_level,
                                                        "biorthogonal", Kokkos::DefaultHostExecutionSpace());
                paliwa::dehierarchize(working, strided_domain, level, minimum_level, maximum_level,
                                                          "biorthogonal", Kokkos::DefaultHostExecutionSpace());

                benchmark::DoNotOptimize(working);
                benchmark::ClobberMemory();
        }

        state.SetItemsProcessed(state.iterations() * (long long)strided_domain.size());
}

// Simple wrappers for 5D and 6D using 6 and 7 dims respectively by composing types
static void BM_HierarchizeDehierarchizeRoundtrip5D(benchmark::State& state)
{
        using SDDom = ddc::StridedDiscreteDomain<BDim0, BDim1, BDim2, BDimX, BDimY>;
        using DElem = ddc::DiscreteElement<BDim0, BDim1, BDim2, BDimX, BDimY>;
        using DVect = ddc::DiscreteVector<BDim0, BDim1, BDim2, BDimX, BDimY>;

        long int const level_val = state.range(0);
        DVect const level(std::array<long int,5>{level_val, level_val, level_val, level_val, level_val});
        DVect const minimum_level(std::array<long int,5>{1,1,1,1,1});
        DVect const maximum_level(std::array<long int,5>{level_val+1, level_val+1, level_val+1, level_val+1, level_val+1});

        SDDom const strided_domain = paliwa::strided_domain_from_level<BDim0,BDim1,BDim2,BDimX,BDimY>(
                        ddc::detail::array(level), ddc::detail::array(maximum_level));

        ddc::Chunk baseline_chunk("baseline5d", strided_domain, ddc::HostAllocator<double>());
        ddc::Chunk working_chunk("working5d", strided_domain, ddc::HostAllocator<double>());
        auto baseline = baseline_chunk.span_view();
        auto working = working_chunk.span_view();

        fill_reference_data(baseline, strided_domain);

        for (auto _ : state) {
                state.PauseTiming();
                ddc::host_for_each(strided_domain, [&](DElem const ixy) { working(ixy) = baseline(ixy); });
                state.ResumeTiming();

                paliwa::hierarchize(working, strided_domain, level, minimum_level, maximum_level,
                                                        "biorthogonal", Kokkos::DefaultHostExecutionSpace());
                paliwa::dehierarchize(working, strided_domain, level, minimum_level, maximum_level,
                                                          "biorthogonal", Kokkos::DefaultHostExecutionSpace());

                benchmark::DoNotOptimize(working);
                benchmark::ClobberMemory();
        }

        state.SetItemsProcessed(state.iterations() * (long long)strided_domain.size());
}

static void BM_HierarchizeDehierarchizeRoundtrip6D(benchmark::State& state)
{
        using SDDom = ddc::StridedDiscreteDomain<BDim0, BDim1, BDim2, BDimX, BDimY, BDimZ>;
        using DElem = ddc::DiscreteElement<BDim0, BDim1, BDim2, BDimX, BDimY, BDimZ>;
        using DVect = ddc::DiscreteVector<BDim0, BDim1, BDim2, BDimX, BDimY, BDimZ>;

        long int const level_val = state.range(0);
        DVect const level(std::array<long int,6>{level_val, level_val, level_val, level_val, level_val, level_val});
        DVect const minimum_level(std::array<long int,6>{1,1,1,1,1,1});
        DVect const maximum_level(std::array<long int,6>{level_val+1, level_val+1, level_val+1, level_val+1, level_val+1, level_val+1});

        SDDom const strided_domain = paliwa::strided_domain_from_level<BDim0,BDim1,BDim2,BDimX,BDimY,BDimZ>(
                        ddc::detail::array(level), ddc::detail::array(maximum_level));

        ddc::Chunk baseline_chunk("baseline6d", strided_domain, ddc::HostAllocator<double>());
        ddc::Chunk working_chunk("working6d", strided_domain, ddc::HostAllocator<double>());
        auto baseline = baseline_chunk.span_view();
        auto working = working_chunk.span_view();

        fill_reference_data(baseline, strided_domain);

        for (auto _ : state) {
                state.PauseTiming();
                ddc::host_for_each(strided_domain, [&](DElem const ixy) { working(ixy) = baseline(ixy); });
                state.ResumeTiming();

                paliwa::hierarchize(working, strided_domain, level, minimum_level, maximum_level,
                                                        "biorthogonal", Kokkos::DefaultHostExecutionSpace());
                paliwa::dehierarchize(working, strided_domain, level, minimum_level, maximum_level,
                                                          "biorthogonal", Kokkos::DefaultHostExecutionSpace());

                benchmark::DoNotOptimize(working);
                benchmark::ClobberMemory();
        }

        state.SetItemsProcessed(state.iterations() * (long long)strided_domain.size());
}

BENCHMARK(BM_RestrictStridedWithDiscrete2D)
        ->Args({8, 8})
        ->Args({10, 10})
        ->Args({12, 12});

BENCHMARK(BM_HierarchizeDehierarchizeRoundtrip2D)
        ->Arg(7)
        ->Arg(8)
        ->Arg(9);

BENCHMARK(BM_HierarchizeDehierarchizeRoundtrip3D)
        ->Arg(5)
        ->Arg(6)
        ->Arg(7);

BENCHMARK(BM_HierarchizeDehierarchizeRoundtrip4D)
        ->Arg(4)
        ->Arg(5);

BENCHMARK(BM_HierarchizeDehierarchizeRoundtrip5D)
        ->Arg(3)
        ->Arg(4);

BENCHMARK(BM_HierarchizeDehierarchizeRoundtrip6D)
        ->Arg(2)
        ->Arg(3);
