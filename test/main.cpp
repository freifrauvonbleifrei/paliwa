// SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: LGPL-2.1-or-later

#include <Kokkos_Core.hpp>
#include <ddc/ddc.hpp>
#include <gtest/gtest.h>

#include "../src/paliwa_distribute.hpp"

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  {
    paliwa::MPIOptionalGuard const mpi_guard(argc, argv);
    Kokkos::ScopeGuard const kokkos_scope(argc, argv);
    ddc::ScopeGuard const ddc_scope(argc, argv);
    return RUN_ALL_TESTS();
  }
}