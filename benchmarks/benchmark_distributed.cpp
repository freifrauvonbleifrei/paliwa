// SPDX-FileCopyrightText: 2026 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "benchmark_common.hpp"

#include "../paliwa/paliwa_dimensions.hpp"
#include "../paliwa/paliwa_domains.hpp"
#ifdef PALIWA_WITH_MPI
#include "../paliwa/paliwa_distributed_transform.hpp"
#endif

namespace paliwa_benchmarks {

#ifdef PALIWA_WITH_MPI
constexpr int kDistributedBenchmarkIterations = 50;

template <class... DDims>
long long communication_bytes(
        ddc::StridedDiscreteDomain<DDims...> const &full_strided,
        ddc::DiscreteDomain<DDims...> const &local_domain,
        ddc::DiscreteVector<DDims...> const &level,
        ddc::DiscreteVector<DDims...> const &minimum_level,
        ddc::DiscreteVector<DDims...> const &maximum_level,
        std::string const &wavelet_name,
        MPI_Comm cart_comm) {
    long long total_bytes = 0;
    auto local_restricted = paliwa::restrict_strided_with_discrete(full_strided, local_domain);
    int dim_index = 0;

    auto estimate_dim = [&](auto dim_tag) {
        using Dim = decltype(dim_tag);
        auto full_1d_strided = ddc::select<Dim>(full_strided);
        auto local_1d = ddc::select<Dim>(local_domain);
        auto local_restricted_1d = ddc::select<Dim>(local_restricted);
        auto global_size_1d = full_1d_strided.extents().template get<Dim>() *
                              full_1d_strided.strides().template get<Dim>();
        ddc::DiscreteDomain<Dim> global_domain_1d(
                ddc::DiscreteElement<Dim>(full_1d_strided.front()),
                ddc::DiscreteVector<Dim>(global_size_1d));

        auto ghost_domain_1d = paliwa::get_required_transform_domain<Dim>(
                true, full_1d_strided, local_restricted_1d,
                ddc::select<Dim>(level), ddc::select<Dim>(minimum_level),
                ddc::select<Dim>(maximum_level), wavelet_name);
        auto ghost_by_rank = paliwa::classify_ghost_by_rank<Dim>(
                ghost_domain_1d, global_domain_1d, cart_comm, dim_index);
        auto peers = paliwa::detail::compute_peer_exchanges<Dim, true>(
                std::move(ghost_by_rank), full_1d_strided, global_domain_1d,
                local_1d, level, minimum_level, maximum_level,
                wavelet_name, cart_comm, dim_index);

        long long pole_bases = static_cast<long long>(local_restricted.size()) /
                               static_cast<long long>(local_restricted_1d.size());
        for (auto const &peer : peers) {
            total_bytes += static_cast<long long>(peer.indices_we_need.size() +
                                                  peer.indices_they_need.size()) *
                           pole_bases * static_cast<long long>(sizeof(double));
        }
        ++dim_index;
    };

    (estimate_dim(DDims{}), ...);
    return total_bytes;
}

template <class... DDims>
std::array<int, sizeof...(DDims)> make_rank_layout() {
    std::array<int, sizeof...(DDims)> layout{};
    layout.fill(1);
    for (size_t i = 0; i + 1 < layout.size(); ++i) {
        layout[i] = 2;
    }
    return layout;
}

template <class... DDims>
static void BM_DistributedHierarchize(benchmark::State &state) {
    using SDDom = ddc::StridedDiscreteDomain<DDims...>;
    using DElem = ddc::DiscreteElement<DDims...>;
    using DVect = ddc::DiscreteVector<DDims...>;

    int world_size = 0;
    int world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    auto const par_vector = make_rank_layout<DDims...>();
    int const required_ranks = std::reduce(par_vector.begin(), par_vector.end(), 1, std::multiplies<>());

    if (world_size < required_ranks) {
        state.SkipWithError("Need at least the required MPI ranks for this benchmark");
        return;
    }

    int const color = (world_rank < required_ranks) ? 0 : MPI_UNDEFINED;
    MPI_Comm sub_comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &sub_comm);
    if (color == MPI_UNDEFINED) {
        state.SkipWithError("Benchmark uses only the first participating MPI ranks");
        return;
    }

    long int const level_val = state.range(0);
    DVect const level(std::array<long int, sizeof...(DDims)>{((void)DDims{}, level_val)...});
    DVect const minimum_level(std::array<long int, sizeof...(DDims)>{((void)DDims{}, 1L)...});
    DVect const maximum_level(
            std::array<long int, sizeof...(DDims)>{((void)DDims{}, level_val + 1)...});
    DVect resolution(std::array<long int, sizeof...(DDims)>{((void)DDims{}, 1L << level_val)...});

    auto global_domain = paliwa::optional_initialize_dims_periodic_unit_cube<DDims...>(resolution);
    auto [local_domain, cart_comm] = paliwa::decompose_domain_on_communicator(
            global_domain, sub_comm, par_vector);

    SDDom const full_strided = paliwa::strided_domain_from_level<DDims...>(
            ddc::detail::array(level), ddc::detail::array(maximum_level));
    auto local_strided = paliwa::restrict_strided_with_discrete(full_strided, local_domain);

    ddc::Chunk baseline_chunk("baseline_distributed", local_strided, ddc::HostAllocator<double>());
    ddc::Chunk working_chunk("working_distributed", local_strided, ddc::HostAllocator<double>());
    auto baseline = baseline_chunk.span_view();
    auto working = working_chunk.span_view();

    fill_reference_data(baseline, local_strided);

    for (auto _ : state) {
        state.PauseTiming();
        ddc::host_for_each(local_strided, [&](DElem const idx) { working(idx) = baseline(idx); });
        state.ResumeTiming();

        paliwa::distributed_hierarchize(
                working, full_strided, level, minimum_level, maximum_level,
                "biorthogonal", cart_comm, Kokkos::DefaultHostExecutionSpace());

        benchmark::DoNotOptimize(working);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<long long>(local_strided.size()));
    set_memory_bandwidth_counter(state, local_strided);
        set_working_set_counter(state, local_strided);
    state.counters["ComBW"] = benchmark::Counter(
            static_cast<double>(communication_bytes<DDims...>(
                    full_strided, local_domain, level, minimum_level,
                    maximum_level, "biorthogonal", cart_comm)),
            benchmark::Counter::kIsRate);

    MPI_Comm_free(&cart_comm);
    MPI_Comm_free(&sub_comm);
}

static void BM_DistributedHierarchize2D(benchmark::State &state) {
    BM_DistributedHierarchize<BDimX, BDimY>(state);
}

static void BM_DistributedHierarchize3D(benchmark::State &state) {
    BM_DistributedHierarchize<BDimX, BDimY, BDimZ>(state);
}

static void BM_DistributedHierarchize4D(benchmark::State &state) {
    BM_DistributedHierarchize<BDim0, BDim1, BDim2, BDimX>(state);
}

// --------- Strong and Weak scaling variants ---------
template <class... DDims>
static std::array<int, sizeof...(DDims)> build_par_vector_from_factor(int factor) {
        std::array<int, sizeof...(DDims)> pv{};
        pv.fill(1);
        size_t dims = pv.size();
        for (int k = 0; k < factor; ++k) {
                pv[k % dims] *= 2;
        }
        return pv;
}

template <class... DDims>
static void BM_DistributedHierarchize_Strong(benchmark::State &state) {
        using SDDom = ddc::StridedDiscreteDomain<DDims...>;
        using DElem = ddc::DiscreteElement<DDims...>;
        using DVect = ddc::DiscreteVector<DDims...>;

        int world_size = 0;
        int world_rank = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        long int const level_val = state.range(0);
        int rank_factor = static_cast<int>(state.range(1));
        auto par_vector = build_par_vector_from_factor<DDims...>(rank_factor);
        int const required_ranks = std::reduce(par_vector.begin(), par_vector.end(), 1, std::multiplies<>());

        if (world_size < required_ranks) {
                state.SkipWithError("Need at least the required MPI ranks for this benchmark");
                return;
        }

        int const color = (world_rank < required_ranks) ? 0 : MPI_UNDEFINED;
        MPI_Comm sub_comm = MPI_COMM_NULL;
        MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &sub_comm);
        if (color == MPI_UNDEFINED) {
                state.SkipWithError("Benchmark uses only the first participating MPI ranks");
                return;
        }

        DVect const level(std::array<long int, sizeof...(DDims)>{((void)DDims{}, level_val)...});
        DVect const minimum_level(std::array<long int, sizeof...(DDims)>{((void)DDims{}, 1L)...});
        DVect const maximum_level(std::array<long int, sizeof...(DDims)>{((void)DDims{}, level_val + 1)...});

        std::array<long int, sizeof...(DDims)> res_arr{};
        for (size_t i = 0; i < res_arr.size(); ++i) res_arr[i] = (1L << level_val);
        DVect resolution_dv(res_arr);
        auto global_domain = paliwa::optional_initialize_dims_periodic_unit_cube<DDims...>(resolution_dv);

        auto [local_domain, cart_comm] = paliwa::decompose_domain_on_communicator(
                        global_domain, sub_comm, par_vector);

        SDDom const full_strided = paliwa::strided_domain_from_level<DDims...>(
                        ddc::detail::array(level), ddc::detail::array(maximum_level));
        auto local_strided = paliwa::restrict_strided_with_discrete(full_strided, local_domain);

        ddc::Chunk baseline_chunk("baseline_distributed", local_strided, ddc::HostAllocator<double>());
        ddc::Chunk working_chunk("working_distributed", local_strided, ddc::HostAllocator<double>());
        auto baseline = baseline_chunk.span_view();
        auto working = working_chunk.span_view();

        fill_reference_data(baseline, local_strided);

        for (auto _ : state) {
                state.PauseTiming();
                ddc::host_for_each(local_strided, [&](DElem const idx) { working(idx) = baseline(idx); });
                state.ResumeTiming();

                paliwa::distributed_hierarchize(
                                working, full_strided, level, minimum_level, maximum_level,
                                "biorthogonal", cart_comm, Kokkos::DefaultHostExecutionSpace());

                benchmark::DoNotOptimize(working);
                benchmark::ClobberMemory();
        }

        state.SetItemsProcessed(state.iterations() * static_cast<long long>(local_strided.size()));
        set_memory_bandwidth_counter(state, local_strided);
        set_working_set_counter(state, local_strided);
        state.counters["ComBW"] = benchmark::Counter(
                        static_cast<double>(communication_bytes<DDims...>(
                                        full_strided, local_domain, level, minimum_level,
                                        maximum_level, "biorthogonal", cart_comm)),
                        benchmark::Counter::kIsRate);

        MPI_Comm_free(&cart_comm);
        MPI_Comm_free(&sub_comm);
}

template <class... DDims>
static void BM_DistributedHierarchize_Weak(benchmark::State &state) {
        using SDDom = ddc::StridedDiscreteDomain<DDims...>;
        using DElem = ddc::DiscreteElement<DDims...>;
        using DVect = ddc::DiscreteVector<DDims...>;

        int world_size = 0;
        int world_rank = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        long int const local_level = state.range(0);
        int rank_factor = static_cast<int>(state.range(1));
        auto par_vector = build_par_vector_from_factor<DDims...>(rank_factor);
        int const required_ranks = std::reduce(par_vector.begin(), par_vector.end(), 1, std::multiplies<>());

        if (world_size < required_ranks) {
                state.SkipWithError("Need at least the required MPI ranks for this benchmark");
                return;
        }

        int const color = (world_rank < required_ranks) ? 0 : MPI_UNDEFINED;
        MPI_Comm sub_comm = MPI_COMM_NULL;
        MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &sub_comm);
        if (color == MPI_UNDEFINED) {
                state.SkipWithError("Benchmark uses only the first participating MPI ranks");
                return;
        }

        // Build global resolution as per-rank resolution * par_vector
        std::array<long int, sizeof...(DDims)> resolution{};
        for (size_t i = 0; i < resolution.size(); ++i) {
                resolution[i] = (1L << local_level) * static_cast<long int>(par_vector[i]);
        }

        DVect const level(std::array<long int, sizeof...(DDims)>{((void)DDims{}, local_level)...});
        DVect const minimum_level(std::array<long int, sizeof...(DDims)>{((void)DDims{}, 1L)...});
        DVect const maximum_level(std::array<long int, sizeof...(DDims)>{((void)DDims{}, local_level + 1)...});

        DVect resolution_dv(resolution);
        auto global_domain = paliwa::optional_initialize_dims_periodic_unit_cube<DDims...>(resolution_dv);
        auto [local_domain, cart_comm] = paliwa::decompose_domain_on_communicator(
                        global_domain, sub_comm, par_vector);

        SDDom const full_strided = paliwa::strided_domain_from_level<DDims...>(
                        ddc::detail::array(level), ddc::detail::array(maximum_level));
        auto local_strided = paliwa::restrict_strided_with_discrete(full_strided, local_domain);

        ddc::Chunk baseline_chunk("baseline_distributed", local_strided, ddc::HostAllocator<double>());
        ddc::Chunk working_chunk("working_distributed", local_strided, ddc::HostAllocator<double>());
        auto baseline = baseline_chunk.span_view();
        auto working = working_chunk.span_view();

        fill_reference_data(baseline, local_strided);

        for (auto _ : state) {
                state.PauseTiming();
                ddc::host_for_each(local_strided, [&](DElem const idx) { working(idx) = baseline(idx); });
                state.ResumeTiming();

                paliwa::distributed_hierarchize(
                                working, full_strided, level, minimum_level, maximum_level,
                                "biorthogonal", cart_comm, Kokkos::DefaultHostExecutionSpace());

                benchmark::DoNotOptimize(working);
                benchmark::ClobberMemory();
        }

        state.SetItemsProcessed(state.iterations() * static_cast<long long>(local_strided.size()));
        set_memory_bandwidth_counter(state, local_strided);
        set_working_set_counter(state, local_strided);
        state.counters["ComBW"] = benchmark::Counter(
                        static_cast<double>(communication_bytes<DDims...>(
                                        full_strided, local_domain, level, minimum_level,
                                        maximum_level, "biorthogonal", cart_comm)),
                        benchmark::Counter::kIsRate);

        MPI_Comm_free(&cart_comm);
        MPI_Comm_free(&sub_comm);
}

static void BM_DistributedHierarchize_Strong2D(benchmark::State &state) { BM_DistributedHierarchize_Strong<BDimX, BDimY>(state); }
static void BM_DistributedHierarchize_Strong4D(benchmark::State &state) { BM_DistributedHierarchize_Strong<BDim0, BDim1, BDim2, BDimX>(state); }
static void BM_DistributedHierarchize_Strong6D(benchmark::State &state) { BM_DistributedHierarchize_Strong<BDim0, BDim1, BDim2, BDimX, BDimY, BDimZ>(state); }

static void BM_DistributedHierarchize_Weak2D(benchmark::State &state) { BM_DistributedHierarchize_Weak<BDimX, BDimY>(state); }
static void BM_DistributedHierarchize_Weak4D(benchmark::State &state) { BM_DistributedHierarchize_Weak<BDim0, BDim1, BDim2, BDimX>(state); }
static void BM_DistributedHierarchize_Weak6D(benchmark::State &state) { BM_DistributedHierarchize_Weak<BDim0, BDim1, BDim2, BDimX, BDimY, BDimZ>(state); }

#endif

} // namespace paliwa_benchmarks

#ifdef PALIWA_WITH_MPI
BENCHMARK(paliwa_benchmarks::BM_DistributedHierarchize2D)
        ->Iterations(paliwa_benchmarks::kDistributedBenchmarkIterations)
        ->Arg(6)
        ->Arg(7)
        ->Arg(8)
        ->Arg(9)
        ->Arg(10);

BENCHMARK(paliwa_benchmarks::BM_DistributedHierarchize3D)
        ->Iterations(paliwa_benchmarks::kDistributedBenchmarkIterations)
        ->Arg(4)
        ->Arg(5)
        ->Arg(6)
        ->Arg(7)
        ->Arg(8);

BENCHMARK(paliwa_benchmarks::BM_DistributedHierarchize4D)
        ->Iterations(paliwa_benchmarks::kDistributedBenchmarkIterations)
        ->Arg(3)
        ->Arg(4)
        ->Arg(5)
        ->Arg(6);
#endif

// Strong-scaling registrations (level, rank_factor)
#ifdef PALIWA_WITH_MPI
BENCHMARK(paliwa_benchmarks::BM_DistributedHierarchize_Strong2D)
        ->Iterations(paliwa_benchmarks::kDistributedBenchmarkIterations)
        ->Args({6, 0})->Args({7, 0})->Args({8, 0})->Args({9, 0})->Args({10, 0})
        ->Args({7, 1})->Args({7, 2});

BENCHMARK(paliwa_benchmarks::BM_DistributedHierarchize_Strong4D)
        ->Iterations(paliwa_benchmarks::kDistributedBenchmarkIterations)
        ->Args({3, 0})->Args({4, 0})->Args({5, 0})->Args({6, 0})
        ->Args({4, 1})->Args({4, 2})->Args({4, 3});

BENCHMARK(paliwa_benchmarks::BM_DistributedHierarchize_Strong6D)
        ->Iterations(paliwa_benchmarks::kDistributedBenchmarkIterations)
        ->Args({1, 0})->Args({2, 0})->Args({3, 0})->Args({4, 0})
        ->Args({2, 1})->Args({2, 2})->Args({2, 3});

// Weak-scaling registrations (local_level, rank_factor)
BENCHMARK(paliwa_benchmarks::BM_DistributedHierarchize_Weak2D)
        ->Iterations(paliwa_benchmarks::kDistributedBenchmarkIterations)
        ->Args({5, 0})->Args({6, 0})->Args({6, 1})->Args({6, 2});

BENCHMARK(paliwa_benchmarks::BM_DistributedHierarchize_Weak4D)
        ->Iterations(paliwa_benchmarks::kDistributedBenchmarkIterations)
        ->Args({2, 0})->Args({3, 0})->Args({3, 1})->Args({3, 2})->Args({3, 3});

BENCHMARK(paliwa_benchmarks::BM_DistributedHierarchize_Weak6D)
        ->Iterations(paliwa_benchmarks::kDistributedBenchmarkIterations)
        ->Args({1, 0})->Args({2, 0})->Args({2, 1})->Args({2, 2})->Args({2, 3});
#endif
