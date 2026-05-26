// SPDX-FileCopyrightText: 2026 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>
#include <ddc/ddc.hpp>

#include "../paliwa/paliwa_distribute.hpp"

int main(int argc, char** argv)
{
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }

    [[maybe_unused]] paliwa::MPIOptionalGuard const mpi(argc, argv);
    Kokkos::ScopeGuard const kokkos_scope(argc, argv);
    ddc::ScopeGuard const ddc_scope(argc, argv);

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
