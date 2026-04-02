// SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <cmath>

#ifdef PALIWA_WITH_MPI
#include <mpi.h>
#endif

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <ddc/ddc.hpp>

#include "../paliwa/paliwa_dimensions.hpp"
#include "../paliwa/paliwa_domains.hpp"
#include "../paliwa/paliwa_transform.hpp"

constexpr double pi_dist = 3.14159265358979323846;

void test_get_rank_local_domain_along_dim() {
  using Dim = paliwa::DDimV;
  auto global_dom = paliwa::initialize_dim_periodic_unit_interval<Dim>(
      ddc::DiscreteVector<Dim>(128));

  auto r0 = paliwa::get_rank_local_domain_along_dim<Dim>(global_dom, 4, 0);
  auto r1 = paliwa::get_rank_local_domain_along_dim<Dim>(global_dom, 4, 1);
  auto r2 = paliwa::get_rank_local_domain_along_dim<Dim>(global_dom, 4, 2);
  auto r3 = paliwa::get_rank_local_domain_along_dim<Dim>(global_dom, 4, 3);

  EXPECT_EQ(r0.size(), 32u);
  EXPECT_EQ(r1.size(), 32u);
  EXPECT_EQ(r0.front().uid<Dim>(), 0);
  EXPECT_EQ(r1.front().uid<Dim>(), 32);
  EXPECT_EQ(r2.front().uid<Dim>(), 64);
  EXPECT_EQ(r3.front().uid<Dim>(), 96);

  auto h0 = paliwa::get_rank_local_domain_along_dim<Dim>(global_dom, 2, 0);
  auto h1 = paliwa::get_rank_local_domain_along_dim<Dim>(global_dom, 2, 1);
  EXPECT_EQ(h0.size(), 64u);
  EXPECT_EQ(h1.size(), 64u);
  EXPECT_EQ(h0.front().uid<Dim>(), 0);
  EXPECT_EQ(h1.front().uid<Dim>(), 64);
}

TEST(distribute, get_rank_local_domain_along_dim) {
  test_get_rank_local_domain_along_dim();
}

void test_classify_ghost_by_coord() {
  using Dim = paliwa::DDimV;
  auto global_dom = ddc::DiscreteDomain<Dim>(ddc::DiscreteElement<Dim>(0),
                                             ddc::DiscreteVector<Dim>(128));

  using DElem = ddc::DiscreteElement<Dim>;

  Kokkos::View<DElem *, Kokkos::SharedSpace> g1("g1", 3);
  g1(0) = DElem(64);
  g1(1) = DElem(66);
  g1(2) = DElem(126);
  auto c1 = paliwa::classify_ghost_by_coord<Dim>(
      ddc::SparseDiscreteDomain<Dim>(g1), global_dom, 2);
  EXPECT_EQ(c1.size(), 1u);
  ASSERT_EQ(c1.count(1), 1u);
  EXPECT_EQ(c1.at(1).size(), 3u);

  Kokkos::View<DElem *, Kokkos::SharedSpace> g2("g2", 3);
  g2(0) = DElem(0);
  g2(1) = DElem(2);
  g2(2) = DElem(62);
  auto c2 = paliwa::classify_ghost_by_coord<Dim>(
      ddc::SparseDiscreteDomain<Dim>(g2), global_dom, 2);
  EXPECT_EQ(c2.size(), 1u);
  ASSERT_EQ(c2.count(0), 1u);
  EXPECT_EQ(c2.at(0).size(), 3u);

  Kokkos::View<DElem *, Kokkos::SharedSpace> g4("g4", 4);
  g4(0) = DElem(10);
  g4(1) = DElem(40);
  g4(2) = DElem(70);
  g4(3) = DElem(100);
  auto c4 = paliwa::classify_ghost_by_coord<Dim>(
      ddc::SparseDiscreteDomain<Dim>(g4), global_dom, 4);
  EXPECT_EQ(c4.size(), 4u);
  for (int c = 0; c < 4; ++c) {
    ASSERT_EQ(c4.count(c), 1u);
    EXPECT_EQ(c4.at(c).size(), 1u);
  }
}

TEST(distribute, classify_ghost_by_coord) { test_classify_ghost_by_coord(); }

#ifdef PALIWA_WITH_MPI

void test_classify_ghost_by_rank() {
  using Dim = paliwa::DDimW;

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (world_size < 2) {
    GTEST_SKIP() << "Need at least 2 ranks";
  }

  int color = (world_rank < 2) ? 0 : MPI_UNDEFINED;
  MPI_Comm sub_comm;
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &sub_comm);
  if (color == MPI_UNDEFINED)
    return;

  auto global_dom = paliwa::initialize_dim_periodic_unit_interval<Dim>(
      ddc::DiscreteVector<Dim>(128));

  std::array<int, 1> par_vector = {2};
  auto [local_dom, cart_comm] = paliwa::decompose_domain_on_communicator(
      global_dom, sub_comm, par_vector);

  using DElem = ddc::DiscreteElement<Dim>;
  Kokkos::View<DElem *, Kokkos::SharedSpace> ghost_elems("ghost", 3);
  if (world_rank == 0) {
    ghost_elems(0) = DElem(64);
    ghost_elems(1) = DElem(66);
    ghost_elems(2) = DElem(126);
  } else {
    ghost_elems(0) = DElem(0);
    ghost_elems(1) = DElem(2);
    ghost_elems(2) = DElem(62);
  }

  auto classified = paliwa::classify_ghost_by_rank<Dim>(
      ddc::SparseDiscreteDomain<Dim>(ghost_elems), global_dom, cart_comm, 0);

  EXPECT_EQ(classified.size(), 1u);
  int my_rank;
  MPI_Comm_rank(cart_comm, &my_rank);
  int other = 1 - my_rank;
  ASSERT_EQ(classified.count(other), 1u);
  EXPECT_EQ(classified.at(other).size(), 3u);

  if (my_rank == 0) {
    EXPECT_EQ(classified.at(other)[0].uid<Dim>(), 64);
    EXPECT_EQ(classified.at(other)[1].uid<Dim>(), 66);
    EXPECT_EQ(classified.at(other)[2].uid<Dim>(), 126);
  } else {
    EXPECT_EQ(classified.at(other)[0].uid<Dim>(), 0);
    EXPECT_EQ(classified.at(other)[1].uid<Dim>(), 2);
    EXPECT_EQ(classified.at(other)[2].uid<Dim>(), 62);
  }

  MPI_Comm_free(&cart_comm);
  MPI_Comm_free(&sub_comm);
}

TEST(distributed, classify_ghost_by_rank) { test_classify_ghost_by_rank(); }

#endif // PALIWA_WITH_MPI
