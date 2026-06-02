// SPDX-FileCopyrightText: 2026 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>
#include <ddc/ddc.hpp>

#include <cstdio>

#include <string_view>
#include <vector>

#include "../paliwa/paliwa_distribute.hpp"

namespace {

bool has_benchmark_min_time(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        std::string_view const arg(argv[i]);
        if (arg == "--benchmark_min_time" ||
            arg.rfind("--benchmark_min_time=", 0) == 0) {
            return true;
        }
    }
    return false;
}

} // namespace

int main(int argc, char **argv) {
    static constexpr char default_benchmark_min_time[] = "--benchmark_min_time=5s";

    std::vector<char *> injected_argv;
    if (!has_benchmark_min_time(argc, argv)) {
        injected_argv.reserve(static_cast<size_t>(argc) + 1);
        injected_argv.push_back(argv[0]);
        injected_argv.push_back(const_cast<char *>(default_benchmark_min_time));
        for (int i = 1; i < argc; ++i) {
            injected_argv.push_back(argv[i]);
        }
        argc = static_cast<int>(injected_argv.size());
        argv = injected_argv.data();
    }

    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }

    [[maybe_unused]] paliwa::MPIOptionalGuard const mpi(argc, argv);

#ifdef PALIWA_WITH_MPI
    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_rank != 0) {
        std::freopen("/dev/null", "w", stdout);
        std::freopen("/dev/null", "w", stderr);
    }
#endif

    Kokkos::ScopeGuard const kokkos_scope(argc, argv);
    ddc::ScopeGuard const ddc_scope(argc, argv);

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
