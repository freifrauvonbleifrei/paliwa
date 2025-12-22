// SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-2.1-or-later

// Some parts reused from ddc/tests/main.cpp
// for these parts:
// Copyright (C) The DDC development team, see DDC's COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>
#include <ddc/ddc.hpp>
#include <gtest/gtest.h>

#include "../src/paliwa_distribute.hpp"

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  {
    [[maybe_unused]] paliwa::MPIOptionalGuard const mpi(argc, argv);
    Kokkos::ScopeGuard const kokkos_scope(argc, argv);
#ifdef PALIWA_WITH_MPI
#ifndef NDEBUG
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    for (int i = 0; i < world_size; ++i) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (i == world_rank) {
        std::cout << "paliwa rank " << world_rank << " : device_id "
                  << Kokkos::device_id() << std::endl;
        // Kokkos::print_configuration(std::cout);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif // not NDEBUG
#endif // PALIWA_WITH_MPI
    ddc::ScopeGuard const ddc_scope(argc, argv);
    return RUN_ALL_TESTS();
  }
}