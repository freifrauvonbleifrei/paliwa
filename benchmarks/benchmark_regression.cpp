// SPDX-FileCopyrightText: 2026 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "benchmark_common.hpp"

#include <algorithm>

#include "../paliwa/paliwa_domains.hpp"
#include "../paliwa/paliwa_transform.hpp"

namespace paliwa_benchmarks {

template <class... DDims>
static std::array<long int, sizeof...(DDims)> make_uniform_array(long int value) {
    std::array<long int, sizeof...(DDims)> values{};
    values.fill(value);
    return values;
}

template <class... DDims>
static ddc::DiscreteDomain<DDims...> make_uniform_local_domain(long int extent_xy) {
    using DDom = ddc::DiscreteDomain<DDims...>;

    long int const local_start = std::max(1L, extent_xy / 4L);
    long int const local_extent = std::max(2L, extent_xy / 2L);

    return DDom(ddc::DiscreteDomain<DDims>(
                        ddc::DiscreteElement<DDims>(local_start),
                        ddc::DiscreteVector<DDims>(local_extent))...);
}

template <class... DDims>
static void BM_RestrictStridedWith(benchmark::State &state) {
    using SDDom = ddc::StridedDiscreteDomain<DDims...>;
    using DDom = ddc::DiscreteDomain<DDims...>;

    long int const extent_xy = state.range(0);
    SDDom const strided_domain = make_strided_domain_from_extent<DDims...>(extent_xy);
    DDom const local_domain = make_uniform_local_domain<DDims...>(extent_xy);

    for (auto _ : state) {
        auto restricted = paliwa::restrict_strided_with_discrete(strided_domain, local_domain);
        benchmark::DoNotOptimize(restricted);
    }

    state.SetItemsProcessed(state.iterations());
}

template <class... DDims>
static void BM_StridedDomainFromLevel(benchmark::State &state) {
    std::array<long int, sizeof...(DDims)> const level = make_uniform_array<DDims...>(6L);
    std::array<long int, sizeof...(DDims)> const finest_level = make_uniform_array<DDims...>(7L);

    for (auto _ : state) {
        auto domain = paliwa::strided_domain_from_level<DDims...>(level, finest_level);
        benchmark::DoNotOptimize(domain);
    }

    state.SetItemsProcessed(state.iterations());
}

template <class... DDims>
static void BM_StridedHierarchicalDomainFromLevel(benchmark::State &state) {
    std::array<long int, sizeof...(DDims)> const level = make_uniform_array<DDims...>(6L);
    std::array<long int, sizeof...(DDims)> const finest_level = make_uniform_array<DDims...>(7L);

    for (auto _ : state) {
        auto domain = paliwa::strided_hierarchical_domain_from_level<DDims...>(level, finest_level);
        benchmark::DoNotOptimize(domain);
    }

    state.SetItemsProcessed(state.iterations());
}

template <class... DDims>
static void BM_SparseFromStridedDomain(benchmark::State &state) {
    using SDDom = ddc::StridedDiscreteDomain<DDims...>;

    long int const extent_xy = state.range(0);
    SDDom const strided_domain = make_strided_domain_from_extent<DDims...>(extent_xy);

    for (auto _ : state) {
        auto sparse_domain = paliwa::sparse_from_strided_domain(strided_domain);
        benchmark::DoNotOptimize(sparse_domain);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<long long>(strided_domain.size()));
}

template <class... DDims>
static void BM_RestrictSparseWithOtherDomain(benchmark::State &state) {
    using SDDom = ddc::StridedDiscreteDomain<DDims...>;

    long int const extent_xy = state.range(0);
    SDDom const strided_domain = make_strided_domain_from_extent<DDims...>(extent_xy);
    auto sparse_domain = paliwa::sparse_from_strided_domain(strided_domain);
    auto local_domain = make_uniform_local_domain<DDims...>(extent_xy);

    for (auto _ : state) {
        auto restricted = paliwa::restrict_sparse_with_other_domain(sparse_domain, local_domain);
        benchmark::DoNotOptimize(restricted);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<long long>(sparse_domain.size()));
}

static void BM_RestrictStridedWithDiscrete2D(benchmark::State &state) {
    BM_RestrictStridedWith<BDimX, BDimY>(state);
}

static void BM_RestrictStridedWithDiscrete4D(benchmark::State &state) {
    BM_RestrictStridedWith<BDim0, BDim1, BDim2, BDim3>(state);
}

static void BM_RestrictStridedWithDiscrete6D(benchmark::State &state) {
    BM_RestrictStridedWith<BDim0, BDim1, BDim2, BDim3, BDim4, BDim5>(state);
}

static void BM_StridedDomainFromLevel2D(benchmark::State &state) {
    BM_StridedDomainFromLevel<BDimX, BDimY>(state);
}

static void BM_StridedDomainFromLevel4D(benchmark::State &state) {
    BM_StridedDomainFromLevel<BDim0, BDim1, BDim2, BDim3>(state);
}

static void BM_StridedDomainFromLevel6D(benchmark::State &state) {
    BM_StridedDomainFromLevel<BDim0, BDim1, BDim2, BDim3, BDim4, BDim5>(state);
}

static void BM_StridedHierarchicalDomainFromLevel2D(benchmark::State &state) {
    BM_StridedHierarchicalDomainFromLevel<BDimX, BDimY>(state);
}

static void BM_StridedHierarchicalDomainFromLevel4D(benchmark::State &state) {
    BM_StridedHierarchicalDomainFromLevel<BDim0, BDim1, BDim2, BDim3>(state);
}

static void BM_StridedHierarchicalDomainFromLevel6D(benchmark::State &state) {
    BM_StridedHierarchicalDomainFromLevel<BDim0, BDim1, BDim2, BDim3, BDim4, BDim5>(state);
}

static void BM_SparseFromStridedDomain2D(benchmark::State &state) {
    BM_SparseFromStridedDomain<BDimX, BDimY>(state);
}

static void BM_SparseFromStridedDomain4D(benchmark::State &state) {
    BM_SparseFromStridedDomain<BDim0, BDim1, BDim2, BDim3>(state);
}

static void BM_SparseFromStridedDomain6D(benchmark::State &state) {
    BM_SparseFromStridedDomain<BDim0, BDim1, BDim2, BDim3, BDim4, BDim5>(state);
}

static void BM_RestrictSparseWithOtherDomain2D(benchmark::State &state) {
    BM_RestrictSparseWithOtherDomain<BDimX, BDimY>(state);
}

static void BM_RestrictSparseWithOtherDomain4D(benchmark::State &state) {
    BM_RestrictSparseWithOtherDomain<BDim0, BDim1, BDim2, BDim3>(state);
}

static void BM_RestrictSparseWithOtherDomain6D(benchmark::State &state) {
    BM_RestrictSparseWithOtherDomain<BDim0, BDim1, BDim2, BDim3, BDim4, BDim5>(state);
}

} // namespace paliwa_benchmarks

BENCHMARK(paliwa_benchmarks::BM_RestrictStridedWithDiscrete2D)->Arg(8)->Arg(16)->Arg(32);
BENCHMARK(paliwa_benchmarks::BM_RestrictStridedWithDiscrete4D)->Arg(4)->Arg(8)->Arg(16);
BENCHMARK(paliwa_benchmarks::BM_RestrictStridedWithDiscrete6D)->Arg(4)->Arg(6)->Arg(8);

BENCHMARK(paliwa_benchmarks::BM_StridedDomainFromLevel2D)->Iterations(1000);
BENCHMARK(paliwa_benchmarks::BM_StridedDomainFromLevel4D)->Iterations(1000);
BENCHMARK(paliwa_benchmarks::BM_StridedDomainFromLevel6D)->Iterations(1000);

BENCHMARK(paliwa_benchmarks::BM_StridedHierarchicalDomainFromLevel2D)->Iterations(1000);
BENCHMARK(paliwa_benchmarks::BM_StridedHierarchicalDomainFromLevel4D)->Iterations(1000);
BENCHMARK(paliwa_benchmarks::BM_StridedHierarchicalDomainFromLevel6D)->Iterations(1000);

BENCHMARK(paliwa_benchmarks::BM_SparseFromStridedDomain2D)->Arg(8)->Arg(16)->Arg(32);
BENCHMARK(paliwa_benchmarks::BM_SparseFromStridedDomain4D)->Arg(4)->Arg(8)->Arg(16);
BENCHMARK(paliwa_benchmarks::BM_SparseFromStridedDomain6D)->Arg(4)->Arg(6)->Arg(8);

BENCHMARK(paliwa_benchmarks::BM_RestrictSparseWithOtherDomain2D)->Arg(8)->Arg(16)->Arg(32);
BENCHMARK(paliwa_benchmarks::BM_RestrictSparseWithOtherDomain4D)->Arg(4)->Arg(8)->Arg(16);
BENCHMARK(paliwa_benchmarks::BM_RestrictSparseWithOtherDomain6D)->Arg(4)->Arg(6)->Arg(8);