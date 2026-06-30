// SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <cmath>
#include <numbers>

#ifdef PALIWA_WITH_MPI
#include <mpi.h>
#endif

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <ddc/ddc.hpp>

#include "../paliwa/paliwa_dimensions.hpp"
#include "../paliwa/paliwa_distribute.hpp"
#include "../paliwa/paliwa_domains.hpp"
#include "../paliwa/paliwa_transform.hpp"

constexpr double pi_dist = std::numbers::pi;

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

// ============================================================
// MPI tests
// ============================================================

#ifdef PALIWA_WITH_MPI

// -------------------------------------------------------
// classify_ghost_by_rank
//
// process_group::decompose() is now the decomposition entry point.
// classify_ghost_by_rank() no longer takes a cart_comm argument —
// it reads from process_group internally.
// The sub_comm passed to decompose() creates the cart comm internally;
// we no longer hold or free a cart_comm handle here.
// -------------------------------------------------------
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

  // decompose() creates the Cartesian communicator and stores it in
  // process_group. We only keep the local subdomain.
  auto local_dom =
      paliwa::decompose(global_dom, sub_comm, std::array<int, 1>{2});

  using DElem = ddc::DiscreteElement<Dim>;
  Kokkos::View<DElem *, Kokkos::SharedSpace> ghost_elems("ghost", 3);

  int my_cart_rank = paliwa::process_group::my_rank();
  if (my_cart_rank == 0) {
    ghost_elems(0) = DElem(64);
    ghost_elems(1) = DElem(66);
    ghost_elems(2) = DElem(126);
  } else {
    ghost_elems(0) = DElem(0);
    ghost_elems(1) = DElem(2);
    ghost_elems(2) = DElem(62);
  }

  // classify_ghost_by_rank reads process_group::get_cart_comm() internally.
  auto classified = paliwa::classify_ghost_by_rank<Dim>(
      ddc::SparseDiscreteDomain<Dim>(ghost_elems), global_dom, 0);

  EXPECT_EQ(classified.size(), 1u);
  int other = 1 - my_cart_rank;
  ASSERT_EQ(classified.count(other), 1u);
  EXPECT_EQ(classified.at(other).size(), 3u);

  if (my_cart_rank == 0) {
    EXPECT_EQ(classified.at(other)[0].uid<Dim>(), 64);
    EXPECT_EQ(classified.at(other)[1].uid<Dim>(), 66);
    EXPECT_EQ(classified.at(other)[2].uid<Dim>(), 126);
  } else {
    EXPECT_EQ(classified.at(other)[0].uid<Dim>(), 0);
    EXPECT_EQ(classified.at(other)[1].uid<Dim>(), 2);
    EXPECT_EQ(classified.at(other)[2].uid<Dim>(), 62);
  }

  MPI_Comm_free(&sub_comm);
}

TEST(distributed, classify_ghost_by_rank) { test_classify_ghost_by_rank(); }

// -------------------------------------------------------
// 1-D roundtrip, 2 ranks
//
// Changes from old version:
//   - decompose_domain_on_communicator → paliwa::decompose()
//   - distributed_hierarchize/dehierarchize →
//     paliwa::hierarchize/dehierarchize (unified call, no cart_comm
//     argument)
//   - Serial reference path uses paliwa::hierarchize with full_strided
//     (the new unified signature requires full_strided explicitly)
//   - MPI_Comm_free(&cart_comm) removed — process_group owns it
// -------------------------------------------------------
void test_distributed_roundtrip_1d(std::string const &wavelet_name) {
  using Dim = paliwa::DDimX;

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (world_size < 2) {
    GTEST_SKIP() << "Need at least 2 MPI ranks";
  }

  int color = (world_rank < 2) ? 0 : MPI_UNDEFINED;
  MPI_Comm sub_comm;
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &sub_comm);
  if (color == MPI_UNDEFINED)
    return;

  constexpr long int max_level = 7, level_val = 6, min_level = 1;
  long int global_size = 1L << max_level;

  ddc::DiscreteDomain<Dim> global_dom;
  if (ddc::is_discrete_space_initialized<Dim>()) {
    global_dom = ddc::DiscreteDomain<Dim>(
        ddc::DiscreteElement<Dim>(0), ddc::DiscreteVector<Dim>(global_size));
  } else {
    global_dom = paliwa::initialize_dim_periodic_unit_interval<Dim>(
        ddc::DiscreteVector<Dim>(global_size));
  }

  using DVect = ddc::DiscreteVector<Dim>;
  DVect level_v(std::array<long int, 1>{level_val});
  DVect min_level_v(std::array<long int, 1>{min_level});
  DVect max_level_v(std::array<long int, 1>{max_level});

  // decompose() stores the Cartesian communicator in process_group.
  auto local_dom =
      paliwa::decompose(global_dom, sub_comm, std::array<int, 1>{2});

  auto full_strided = paliwa::strided_domain_from_level<Dim>(
      ddc::detail::array(level_v), ddc::detail::array(max_level_v));
  auto local_strided =
      paliwa::restrict_strided_with_discrete(full_strided, local_dom);

  ddc::Chunk local_chunk("local", local_strided,
                          ddc::HostAllocator<double>());
  auto local_span = local_chunk.span_view();
  ddc::Chunk orig_chunk("orig", local_strided, ddc::HostAllocator<double>());
  auto orig_span = orig_chunk.span_view();

  ddc::host_for_each(local_strided, [&](ddc::DiscreteElement<Dim> elem) {
    local_span(elem) = std::sin(2.0 * pi_dist * ddc::coordinate(elem));
    orig_span(elem) = local_span(elem);
  });

  // Build a serial reference on rank 0 by gathering all data.
  ddc::Chunk serial_chunk("serial", full_strided,
                           ddc::HostAllocator<double>());
  auto serial_span = serial_chunk.span_view();
  {
    int n = static_cast<int>(local_strided.size());
    std::vector<double> vals(n);
    std::size_t idx = 0;
    ddc::host_for_each(local_strided, [&](ddc::DiscreteElement<Dim> e) {
      vals[idx++] = local_span(e);
    });
    if (world_rank == 0) {
      idx = 0;
      ddc::host_for_each(local_strided, [&](ddc::DiscreteElement<Dim> e) {
        serial_span(e) = vals[idx++];
      });
      std::vector<double> remote(n);
      MPI_Recv(remote.data(), n, MPI_DOUBLE, 1, 99, sub_comm,
               MPI_STATUS_IGNORE);
      idx = 0;
      ddc::host_for_each(
          paliwa::restrict_strided_with_discrete(
              full_strided,
              paliwa::get_rank_local_domain_along_dim<Dim>(global_dom, 2, 1)),
          [&](ddc::DiscreteElement<Dim> e) { serial_span(e) = remote[idx++]; });
    } else {
      MPI_Send(vals.data(), n, MPI_DOUBLE, 0, 99, sub_comm);
    }
  }

  // Serial reference: run the single-process transform on the full grid.
  // hierarchize now always takes full_strided explicitly.
  if (world_rank == 0) {
    paliwa::hierarchize(serial_span, full_strided, level_v, min_level_v,
                         max_level_v, wavelet_name,
                         Kokkos::DefaultHostExecutionSpace());
  }

  // Distributed hierarchize — unified API, no cart_comm argument.
  paliwa::hierarchize(local_span, full_strided, level_v, min_level_v,
                       max_level_v, wavelet_name,
                       Kokkos::DefaultHostExecutionSpace());

  // Compare distributed result with serial reference.
  {
    int n = static_cast<int>(local_strided.size());
    std::vector<double> vals(n);
    std::size_t idx = 0;
    ddc::host_for_each(local_strided, [&](ddc::DiscreteElement<Dim> e) {
      vals[idx++] = local_span(e);
    });
    if (world_rank == 0) {
      idx = 0;
      ddc::host_for_each(local_strided, [&](ddc::DiscreteElement<Dim> e) {
        EXPECT_NEAR(vals[idx++], serial_span(e), 1e-12);
      });
      std::vector<double> remote(n);
      MPI_Recv(remote.data(), n, MPI_DOUBLE, 1, 100, sub_comm,
               MPI_STATUS_IGNORE);
      idx = 0;
      ddc::host_for_each(
          paliwa::restrict_strided_with_discrete(
              full_strided,
              paliwa::get_rank_local_domain_along_dim<Dim>(global_dom, 2, 1)),
          [&](ddc::DiscreteElement<Dim> e) {
            EXPECT_NEAR(remote[idx++], serial_span(e), 1e-12);
          });
    } else {
      MPI_Send(vals.data(), n, MPI_DOUBLE, 0, 100, sub_comm);
    }
  }

  // Distributed dehierarchize — unified API, no cart_comm argument.
  paliwa::dehierarchize(local_span, full_strided, level_v, min_level_v,
                         max_level_v, wavelet_name,
                         Kokkos::DefaultHostExecutionSpace());

  ddc::host_for_each(local_strided, [&](ddc::DiscreteElement<Dim> e) {
    EXPECT_NEAR(local_span(e), orig_span(e), 1e-10);
  });

  // sub_comm was created by us; process_group owns the cart comm.
  MPI_Comm_free(&sub_comm);
}

void roundtrip_1d_hat() { test_distributed_roundtrip_1d("hat"); }
void roundtrip_1d_bio() { test_distributed_roundtrip_1d("biorthogonal"); }
void roundtrip_1d_fw() { test_distributed_roundtrip_1d("fullweighting"); }
TEST(distributed, roundtrip_1d_hat) { roundtrip_1d_hat(); }
TEST(distributed, roundtrip_1d_biorthogonal) { roundtrip_1d_bio(); }
TEST(distributed, roundtrip_1d_fullweighting) { roundtrip_1d_fw(); }

// -------------------------------------------------------
// 1-D roundtrip, 8 ranks
// -------------------------------------------------------
void test_distributed_roundtrip_1d_8ranks(std::string const &wavelet_name) {
  using Dim = paliwa::DDimT;

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (world_size < 8) {
    GTEST_SKIP() << "Need at least 8 MPI ranks";
  }

  int color = (world_rank < 8) ? 0 : MPI_UNDEFINED;
  MPI_Comm sub_comm;
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &sub_comm);
  if (color == MPI_UNDEFINED)
    return;

  constexpr long int max_level = 9, level_val = 8, min_level = 2;
  long int global_size = 1L << max_level; // 512

  ddc::DiscreteDomain<Dim> global_dom;
  if (ddc::is_discrete_space_initialized<Dim>()) {
    global_dom = ddc::DiscreteDomain<Dim>(
        ddc::DiscreteElement<Dim>(0), ddc::DiscreteVector<Dim>(global_size));
  } else {
    global_dom = paliwa::initialize_dim_periodic_unit_interval<Dim>(
        ddc::DiscreteVector<Dim>(global_size));
  }

  using DVect = ddc::DiscreteVector<Dim>;
  DVect level_v(std::array<long int, 1>{level_val});
  DVect min_level_v(std::array<long int, 1>{min_level});
  DVect max_level_v(std::array<long int, 1>{max_level});

  auto local_dom =
      paliwa::decompose(global_dom, sub_comm, std::array<int, 1>{8});

  auto full_strided = paliwa::strided_domain_from_level<Dim>(
      ddc::detail::array(level_v), ddc::detail::array(max_level_v));
  auto local_strided =
      paliwa::restrict_strided_with_discrete(full_strided, local_dom);

  ddc::Chunk local_chunk("local", local_strided,
                          ddc::HostAllocator<double>());
  auto local_span = local_chunk.span_view();
  ddc::Chunk orig_chunk("orig", local_strided, ddc::HostAllocator<double>());
  auto orig_span = orig_chunk.span_view();

  ddc::host_for_each(local_strided, [&](ddc::DiscreteElement<Dim> e) {
    local_span(e) = std::sin(2.0 * pi_dist * ddc::coordinate(e));
    orig_span(e) = local_span(e);
  });

  paliwa::hierarchize(local_span, full_strided, level_v, min_level_v,
                       max_level_v, wavelet_name,
                       Kokkos::DefaultHostExecutionSpace());

  paliwa::dehierarchize(local_span, full_strided, level_v, min_level_v,
                         max_level_v, wavelet_name,
                         Kokkos::DefaultHostExecutionSpace());

  ddc::host_for_each(local_strided, [&](ddc::DiscreteElement<Dim> e) {
    EXPECT_NEAR(local_span(e), orig_span(e), 1e-10);
  });

  MPI_Comm_free(&sub_comm);
}

TEST(distributed, roundtrip_1d_8ranks_hat) {
  test_distributed_roundtrip_1d_8ranks("hat");
}
TEST(distributed, roundtrip_1d_8ranks_biorthogonal) {
  test_distributed_roundtrip_1d_8ranks("biorthogonal");
}
TEST(distributed, roundtrip_1d_8ranks_fullweighting) {
  test_distributed_roundtrip_1d_8ranks("fullweighting");
}

// -------------------------------------------------------
// 2-D roundtrip, 4 ranks (2×2 decomposition)
// -------------------------------------------------------
void distributed_roundtrip_2d_all_wavelets() {
  using DimX = paliwa::DDimY;
  using DimY = paliwa::DDimZ;
  using DElem = ddc::DiscreteElement<DimX, DimY>;
  using DVect = ddc::DiscreteVector<DimX, DimY>;

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (world_size < 4) {
    GTEST_SKIP() << "Need at least 4 ranks";
  }

  int color = (world_rank < 4) ? 0 : MPI_UNDEFINED;
  MPI_Comm sub_comm;
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &sub_comm);
  if (color == MPI_UNDEFINED)
    return;

  DVect resolution(std::array<long int, 2>{1L << 6, 1L << 7});
  auto global_dom =
      paliwa::optional_initialize_dims_periodic_unit_cube(resolution);

  DVect level_v(std::array<long int, 2>{5, 6});
  DVect min_level_v(std::array<long int, 2>{1, 2});
  DVect max_level_v(std::array<long int, 2>{6, 7});

  auto local_dom =
      paliwa::decompose(global_dom, sub_comm, std::array<int, 2>{2, 2});

  auto full_strided = paliwa::strided_domain_from_level<DimX, DimY>(
      ddc::detail::array(level_v), ddc::detail::array(max_level_v));
  auto local_strided =
      paliwa::restrict_strided_with_discrete(full_strided, local_dom);

  for (auto const &wn : {"hat", "biorthogonal", "fullweighting"}) {
    SCOPED_TRACE(wn);
    ddc::Chunk lc("local", local_strided, ddc::HostAllocator<double>());
    auto ls = lc.span_view();
    ddc::Chunk oc("orig", local_strided, ddc::HostAllocator<double>());
    auto os = oc.span_view();

    ddc::host_for_each(local_strided, [&](DElem e) {
      auto c = ddc::coordinate(e).array();
      ls(e) = std::sin(2.0 * pi_dist * c[0]) * std::sin(2.0 * pi_dist * c[1]);
      os(e) = ls(e);
    });

    paliwa::hierarchize(ls, full_strided, level_v, min_level_v, max_level_v,
                         std::string(wn), Kokkos::DefaultHostExecutionSpace());
    paliwa::dehierarchize(ls, full_strided, level_v, min_level_v,
                           max_level_v, std::string(wn),
                           Kokkos::DefaultHostExecutionSpace());

    ddc::host_for_each(local_strided,
                        [&](DElem e) { EXPECT_NEAR(ls(e), os(e), 1e-10); });
  }

  MPI_Comm_free(&sub_comm);
}

TEST(distributed, roundtrip_2d_all_wavelets) {
  distributed_roundtrip_2d_all_wavelets();
}

#endif // PALIWA_WITH_MPI