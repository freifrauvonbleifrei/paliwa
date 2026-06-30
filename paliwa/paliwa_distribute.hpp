// SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>

#include <Kokkos_Core.hpp>
#include <ddc/ddc.hpp>

#ifdef PALIWA_WITH_MPI
#include <mpi.h>
#endif

#include "paliwa_domains.hpp"
#include "paliwa_utils.hpp"

namespace paliwa {

// All code outside this file uses MPICommType so that it compiles
// identically with and without MPI. The macros for MPI_COMM_WORLD /
// MPI_COMM_NULL are only defined when the real MPI header has not
// already supplied them.

#ifdef PALIWA_WITH_MPI
using MPICommType = MPI_Comm;
#else
using MPICommType = void *;
#ifndef MPI_COMM_WORLD
#define MPI_COMM_WORLD nullptr
#endif
#ifndef MPI_COMM_NULL
#define MPI_COMM_NULL nullptr
#endif
#endif // PALIWA_WITH_MPI

struct MPIOptionalGuard {
  MPIOptionalGuard([[maybe_unused]] int &argc,
                    [[maybe_unused]] char **&argv) {
#ifdef PALIWA_WITH_MPI
    MPI_Init(&argc, &argv);
#endif
  }
  ~MPIOptionalGuard() {
#ifdef PALIWA_WITH_MPI
    MPI_Finalize();
#endif
  }
  MPIOptionalGuard(MPIOptionalGuard const &) = delete;
  MPIOptionalGuard &operator=(MPIOptionalGuard const &) = delete;
};

// ============================================================
// process_group
// ============================================================
// Stores the Cartesian communicator together with cached topology
// information (dimensions, coordinates, and coordinate-to-rank lookup).
// The cache is initialized by decompose() and avoids repeated MPI
// topology queries during ghost exchanges.

struct process_group {
  process_group() = delete; // Purely static interface.

  // ----------------------------------------------------------
  // Accessors for the communicator
  // ----------------------------------------------------------

  static MPICommType get_cart_comm() { return get_state().m_cart_comm; }

#ifdef PALIWA_WITH_MPI
  // ----------------------------------------------------------
  // Accessors for the precomputed topology cache
  // ----------------------------------------------------------

  /// Number of Cartesian dimensions.
  static int n_dims() { return get_state().m_n_dims; }

  /// Number of ranks along dimension d.
  static int cart_dim(int d) { return get_state().m_cart_dims[d]; }

  /// This rank's Cartesian coordinate along dimension d.
  static int my_coord(int d) { return get_state().m_my_coords[d]; }

  /// This rank's integer rank in the Cartesian communicator.
  static int my_rank() { return get_state().m_my_rank; }

  /// Rank of the process at Cartesian coordinate coord along dimension d.
  /// O(1) table lookup — no MPI call.
  static int coord_to_rank(int d, int coord) {
    State &s = get_state();
    assert(d >= 0 && d < s.m_n_dims);
    assert(coord >= 0 && coord < s.m_cart_dims[d]);
    return s.m_coord_to_rank[static_cast<std::size_t>(d) *
                                  static_cast<std::size_t>(
                                      s.m_max_ranks_per_dim) +
                              static_cast<std::size_t>(coord)];
  }
#endif // PALIWA_WITH_MPI

private:
  struct State {
    MPICommType m_cart_comm = MPI_COMM_NULL;

#ifdef PALIWA_WITH_MPI
    int m_n_dims = 0;
    int m_my_rank = -1;
    int m_max_ranks_per_dim = 0;
    std::vector<int> m_cart_dims; ///< Size: m_n_dims.
    std::vector<int> m_my_coords; ///< Size: m_n_dims.
    // Flat table: m_coord_to_rank[d * m_max_ranks_per_dim + coord] = rank.
    std::vector<int> m_coord_to_rank;
#endif
  };

  // Singleton.
  static State &get_state() {
    static State s;
    return s;
  }

  static void set_state(MPICommType comm) {
    State &s = get_state();
    s.m_cart_comm = comm;

#ifdef PALIWA_WITH_MPI
    if (comm == MPI_COMM_NULL) {
      s.m_n_dims = 0;
      s.m_my_rank = -1;
      s.m_max_ranks_per_dim = 0;
      s.m_cart_dims.clear();
      s.m_my_coords.clear();
      s.m_coord_to_rank.clear();
      return;
    }

    MPI_Cartdim_get(comm, &s.m_n_dims);

    std::vector<int> periods(s.m_n_dims);
    s.m_cart_dims.resize(s.m_n_dims);
    s.m_my_coords.resize(s.m_n_dims);
    MPI_Cart_get(comm, s.m_n_dims, s.m_cart_dims.data(), periods.data(),
                 s.m_my_coords.data());

    MPI_Comm_rank(comm, &s.m_my_rank);

    // Build the coordinate-to-rank lookup table.
    s.m_max_ranks_per_dim =
        *std::max_element(s.m_cart_dims.begin(), s.m_cart_dims.end());
    s.m_coord_to_rank.assign(
        static_cast<std::size_t>(s.m_n_dims) *
            static_cast<std::size_t>(s.m_max_ranks_per_dim),
        -1);

    for (int d = 0; d < s.m_n_dims; ++d) {
      std::vector<int> coords(s.m_my_coords.begin(), s.m_my_coords.end());
      for (int coord = 0; coord < s.m_cart_dims[d]; ++coord) {
        coords[d] = coord;
        int rank = -1;
        MPI_Cart_rank(comm, coords.data(), &rank);
        s.m_coord_to_rank[static_cast<std::size_t>(d) *
                               static_cast<std::size_t>(
                                   s.m_max_ranks_per_dim) +
                           static_cast<std::size_t>(coord)] = rank;
      }
    }
#endif // PALIWA_WITH_MPI
  }

  // Only decompose() may call set_state().
  template <typename DiscreteDomainType>
  friend DiscreteDomainType
  decompose(DiscreteDomainType const &, MPICommType,
            std::array<int, DiscreteDomainType::rank()> const &);
};

#ifdef PALIWA_WITH_MPI

template <typename T> struct MPIValueType {
  MPI_Datatype type;
  bool owned;

  MPIValueType() {
    if constexpr (std::is_same_v<T, double>) {
      type = MPI_DOUBLE;
      owned = false;
    } else if constexpr (std::is_same_v<T, float>) {
      type = MPI_FLOAT;
      owned = false;
    } else {
      MPI_Type_contiguous(sizeof(T), MPI_BYTE, &type);
      MPI_Type_commit(&type);
      owned = true;
    }
  }

  ~MPIValueType() {
    if (owned)
      MPI_Type_free(&type);
  }

  MPIValueType(MPIValueType const &) = delete;
  MPIValueType &operator=(MPIValueType const &) = delete;

  operator MPI_Datatype() const { return type; }
};

#endif // PALIWA_WITH_MPI

template <class HeadTag, class... Tags>
constexpr ddc::DiscreteDomain<HeadTag, Tags...> distribute_idx_range(
    ddc::DiscreteDomain<HeadTag, Tags...> global_idx_range,
    ddc::DiscreteVector<HeadTag, Tags...> const &par_vector,
    ddc::DiscreteVector<HeadTag, Tags...> const &my_coords) {

  ddc::DiscreteDomain<HeadTag> global_1d =
      ddc::select<HeadTag>(global_idx_range);
  auto n_ranks = ddc::DiscreteVector<HeadTag>(par_vector);
  auto rank_coord = ddc::DiscreteVector<HeadTag>(my_coords);

  if (global_1d.size() % n_ranks != 0)
    throw std::runtime_error(
        "distribute_idx_range: global extent is not evenly divisible "
        "by the number of ranks along this dimension.");

  ddc::DiscreteVector<HeadTag> chunk_size(global_1d.size() / n_ranks);
  ddc::DiscreteElement<HeadTag> start(global_1d.front() +
                                       rank_coord * chunk_size);
  ddc::DiscreteDomain<HeadTag> local_1d(start, chunk_size);

  if constexpr (sizeof...(Tags) == 0) {
    return local_1d;
  } else {
    return ddc::DiscreteDomain<HeadTag, Tags...>(
        local_1d,
        distribute_idx_range(ddc::select<Tags...>(global_idx_range),
                              ddc::select<Tags...>(par_vector),
                              ddc::select<Tags...>(my_coords)));
  }
}

template <typename Dim>
constexpr ddc::DiscreteDomain<Dim> get_rank_local_domain_along_dim(
    ddc::DiscreteDomain<Dim> const &global_domain_1d, int n_ranks,
    int rank_coord) {
  assert(n_ranks > 0);
  assert(rank_coord >= 0 && rank_coord < n_ranks);
  auto global_size = static_cast<long int>(global_domain_1d.size());
  assert(global_size % n_ranks == 0);
  long int chunk_size = global_size / n_ranks;
  ddc::DiscreteElement<Dim> start(
      global_domain_1d.front() +
      ddc::DiscreteVector<Dim>(rank_coord * chunk_size));
  return ddc::DiscreteDomain<Dim>(start, ddc::DiscreteVector<Dim>(chunk_size));
}

/// Classify ghost indices by the coordinate of their owner along one
/// Cartesian dimension. Pure index arithmetic — no MPI required.
///
/// @return Map from owner coordinate (0-based) to ghost element list.
template <typename Dim>
std::map<int, std::vector<ddc::DiscreteElement<Dim>>> classify_ghost_by_coord(
    ddc::SparseDiscreteDomain<Dim> const &ghost_domain,
    ddc::DiscreteDomain<Dim> const &global_domain_1d,
    int n_ranks_along_dim) {
  auto global_size = static_cast<long int>(global_domain_1d.size());
  long int chunk_size = global_size / n_ranks_along_dim;
  long int global_front = global_domain_1d.front().template uid<Dim>();

  std::map<int, std::vector<ddc::DiscreteElement<Dim>>> result;
  ddc::host_for_each(ghost_domain, [&](ddc::DiscreteElement<Dim> elem) {
    long int global_idx = elem.template uid<Dim>() - global_front;
    global_idx = ((global_idx % global_size) + global_size) % global_size;
    int owner_coord = static_cast<int>(global_idx / chunk_size);
    if (owner_coord >= n_ranks_along_dim)
      owner_coord = n_ranks_along_dim - 1;
    result[owner_coord].push_back(elem);
  });
  return result;
}

/// Classify ghost indices by owner MPI rank along a Cartesian dimension.
/// Uses the precomputed coord_to_rank table from process_group — no MPI
/// call is made at runtime.
///
/// Without MPI: always returns an empty map — on a single process there
/// are no remote peers and therefore no ghost data to exchange.
template <typename Dim>
std::map<int, std::vector<ddc::DiscreteElement<Dim>>> classify_ghost_by_rank(
    ddc::SparseDiscreteDomain<Dim> const &ghost_domain,
    ddc::DiscreteDomain<Dim> const &global_domain_1d, int dim_index) {
#ifndef PALIWA_WITH_MPI
  (void)ghost_domain;
  (void)global_domain_1d;
  (void)dim_index;
  return {};
#else
  if (process_group::get_cart_comm() == MPI_COMM_NULL)
    return {};

  // n_ranks_along_dim and the coord-to-rank map come from the cache —
  // no MPI_Cart_get or MPI_Cart_rank call here.
  int n_ranks = process_group::cart_dim(dim_index);

  auto by_coord =
      classify_ghost_by_coord<Dim>(ghost_domain, global_domain_1d, n_ranks);

  std::map<int, std::vector<ddc::DiscreteElement<Dim>>> result;
  for (auto &[coord, indices] : by_coord)
    result[process_group::coord_to_rank(dim_index, coord)] =
        std::move(indices);
  return result;
#endif
}

/// Describes what a single peer needs to send to us and what we need
/// to send to them for one ghost exchange along one dimension.
template <typename DElem1d> struct PeerExchange {
  int remote_rank;
  std::vector<DElem1d> indices_we_need;   ///< Ghost indices we recv from peer.
  std::vector<DElem1d> indices_they_need; ///< Our local indices peer needs.
};

/// Build the peer exchange table for one transform dimension.
///
/// Determines both the ghost data we receive and the local data
/// requested by each remote rank. Returns an empty vector when MPI
/// is disabled.
template <typename DimToTransform, bool IsHierarchization,
          typename LocalDomainType, typename RemoteGhostFn>
std::vector<PeerExchange<ddc::DiscreteElement<DimToTransform>>>
compute_peer_exchanges(
    std::map<int, std::vector<ddc::DiscreteElement<DimToTransform>>>
        ghost_by_rank,
    ddc::StridedDiscreteDomain<DimToTransform> const &full_1d_strided,
    ddc::DiscreteDomain<DimToTransform> const &global_domain_1d,
    LocalDomainType const &local_domain_1d, int dim_index,
    RemoteGhostFn remote_ghost_fn) {
#ifndef PALIWA_WITH_MPI
  (void)ghost_by_rank;
  (void)full_1d_strided;
  (void)global_domain_1d;
  (void)local_domain_1d;
  (void)dim_index;
  (void)remote_ghost_fn;
  return {};
#else
  if (process_group::get_cart_comm() == MPI_COMM_NULL)
    return {};

  using DElem1d = ddc::DiscreteElement<DimToTransform>;

  // All topology data comes from the precomputed cache — zero MPI calls.
  int const n_ranks_along_dim = process_group::cart_dim(dim_index);
  int const my_coord = process_group::my_coord(dim_index);

  // For each remote rank, compute which of our local indices they need.
  std::map<int, std::vector<DElem1d>> they_need_from_us;
  for (int coord = 0; coord < n_ranks_along_dim; ++coord) {
    if (coord == my_coord)
      continue;

    int const remote_rank = process_group::coord_to_rank(dim_index, coord);

    auto remote_local_1d = get_rank_local_domain_along_dim<DimToTransform>(
        global_domain_1d, n_ranks_along_dim, coord);
    auto remote_restricted_1d =
        restrict_strided_with_discrete(full_1d_strided, remote_local_1d);

    // Delegate ghost-zone computation back to the caller via the functor
    // so that distribute remains free of transform/wavelet knowledge.
    auto remote_ghost_1d = remote_ghost_fn(remote_restricted_1d);

    std::vector<DElem1d> needed;
    ddc::host_for_each(remote_ghost_1d, [&](DElem1d elem) {
      if (local_domain_1d.contains(elem))
        needed.push_back(elem);
    });
    if (!needed.empty())
      they_need_from_us[remote_rank] = std::move(needed);
  }

  // Merge what we need and what they need into PeerExchange entries.
  std::set<int> all_peer_ranks;
  for (auto &[rank, _] : ghost_by_rank)
    all_peer_ranks.insert(rank);
  for (auto &[rank, _] : they_need_from_us)
    all_peer_ranks.insert(rank);

  std::vector<PeerExchange<DElem1d>> peers;
  peers.reserve(all_peer_ranks.size());
  for (int remote_rank : all_peer_ranks) {
    PeerExchange<DElem1d> peer;
    peer.remote_rank = remote_rank;
    if (ghost_by_rank.count(remote_rank))
      peer.indices_we_need = std::move(ghost_by_rank[remote_rank]);
    if (they_need_from_us.count(remote_rank))
      peer.indices_they_need = std::move(they_need_from_us[remote_rank]);
    peers.push_back(std::move(peer));
  }
  return peers;
#endif
}

/**
 * @brief Exchange ghost hyperplane slices via non-blocking MPI sends/receives.
 *
 * For each peer, builds MPI hindexed derived types that describe the
 * memory locations of all ghost elements directly in the source /
 * destination spans — no intermediate copy buffers needed.
 * All Irecv calls are posted before all Isend calls to avoid deadlocks
 * when peers have asymmetric neighbour lists. A single MPI_Waitall
 * completes all transfers.
 *
 * The communicator is read from process_group — no MPI_Comm argument.
 *
 * Without MPI this function is a no-op. Because compute_peer_exchanges
 * always returns an empty vector without MPI, peers.size() == 0 and even
 * the loop bodies would be dead code; the explicit early return makes the
 * intent obvious to the compiler.
 *
 * @param peers                Peer exchange table from compute_peer_exchanges.
 * @param local_grid           Source span — data is sent from here.
 * @param extended_span        Destination span — ghost data is received here.
 * @param local_restricted     Local multi-D strided domain (drives slice loop).
 * @param local_restricted_1d  Local 1-D strided domain along transform dim.
 */
template <typename DimToTransform, typename SrcSpanType, typename DstSpanType,
          typename... DDims>
void exchange_ghost_slices(
    std::vector<PeerExchange<ddc::DiscreteElement<DimToTransform>>> const
        &peers,
    SrcSpanType const local_grid, DstSpanType const extended_span,
    ddc::StridedDiscreteDomain<DDims...> const &local_restricted,
    ddc::StridedDiscreteDomain<DimToTransform> const &local_restricted_1d) {
#ifndef PALIWA_WITH_MPI
  (void)peers;
  (void)local_grid;
  (void)extended_span;
  (void)local_restricted;
  (void)local_restricted_1d;
  return;
#else
  using value_type = typename SrcSpanType::element_type;
  using DElem1d = ddc::DiscreteElement<DimToTransform>;

  MPI_Comm cart_comm = process_group::get_cart_comm();
  if (cart_comm == MPI_COMM_NULL)
    return;

  MPIValueType<value_type> mpi_value_type;

  // Byte offset of a multi-D element from the span's data pointer.
  auto byte_offset = [](auto const &span,
                         ddc::DiscreteElement<DDims...> elem) {
    return static_cast<MPI_Aint>(
        (&span(elem) - span.data_handle()) *
        static_cast<std::ptrdiff_t>(sizeof(value_type)));
  };

  // Collect one representative multi-D element per (d-1)-D hyperplane
  // slice: all elements whose DimToTransform coordinate matches the
  // first index of the local 1-D domain.
  auto first_local_1d_elem = local_restricted_1d.front();
  std::vector<ddc::DiscreteElement<DDims...>> pole_bases;
  ddc::host_for_each(local_restricted,
                      [&](ddc::DiscreteElement<DDims...> elem) {
                        if (ddc::select<DimToTransform>(elem) ==
                            first_local_1d_elem)
                          pole_bases.push_back(elem);
                      });

  // Build a flat displacement array: each 1-D index combined with every
  // pole base produces one element of one hyperplane slice.
  auto build_displacements = [&](auto const &span,
                                  std::vector<DElem1d> const &indices_1d) {
    std::vector<MPI_Aint> displacements;
    displacements.reserve(indices_1d.size() * pole_bases.size());
    for (auto const &d_elem : indices_1d)
      for (auto const &pole_base : pole_bases)
        displacements.push_back(byte_offset(
            span, paliwa::replace_dim<DimToTransform>(pole_base, d_elem)));
    return displacements;
  };

  std::vector<MPI_Request> requests;
  std::vector<MPI_Datatype> types_to_free;
  requests.reserve(peers.size() * 2);
  types_to_free.reserve(peers.size() * 2);

  for (auto const &peer : peers) {
    auto recv_displ = build_displacements(extended_span, peer.indices_we_need);
    int recv_count = static_cast<int>(recv_displ.size());
    std::vector<int> recv_blocklens(recv_count, 1);

    MPI_Datatype recv_type;
    MPI_Type_create_hindexed(recv_count, recv_blocklens.data(),
                              recv_displ.data(), mpi_value_type, &recv_type);
    MPI_Type_commit(&recv_type);
    types_to_free.push_back(recv_type);

    MPI_Request req;
    MPI_Irecv(extended_span.data_handle(), 1, recv_type, peer.remote_rank, 20,
               cart_comm, &req);
    requests.push_back(req);
  }

  for (auto const &peer : peers) {
    auto send_displ =
        build_displacements(local_grid, peer.indices_they_need);
    int send_count = static_cast<int>(send_displ.size());
    std::vector<int> send_blocklens(send_count, 1);

    MPI_Datatype send_type;
    MPI_Type_create_hindexed(send_count, send_blocklens.data(),
                              send_displ.data(), mpi_value_type, &send_type);
    MPI_Type_commit(&send_type);
    types_to_free.push_back(send_type);

    MPI_Request req;
    MPI_Isend(local_grid.data_handle(), 1, send_type, peer.remote_rank, 20,
               cart_comm, &req);
    requests.push_back(req);
  }

  std::vector<MPI_Status> statuses(requests.size());
  MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
              statuses.data());

  for (auto &t : types_to_free)
    MPI_Type_free(&t);
#endif
}

// ============================================================
// Section 8 — decompose (primary public API)
// ============================================================

/**
 * @brief Decompose a global domain across MPI ranks and initialize the
 *        process_group communicator and topology cache.
 *
 * This is the single entry point a user calls at startup:
 *
 *   auto local_domain =
 *       paliwa::decompose(global, MPI_COMM_WORLD, {2, 4});
 *
 * Internally a periodic Cartesian communicator is created from comm and
 * par_vector. The communicator and a full topology cache (dims, coords,
 * coordinate-to-rank table) are stored in process_group::State so that
 * all subsequent calls to classify_ghost_by_rank and
 * compute_peer_exchanges require zero MPI calls at runtime.
 *
 * Without MPI, par_vector must be all-ones; the global domain is returned
 * unchanged and the stored communicator is nullptr.
 */
template <typename DiscreteDomainType>
DiscreteDomainType
decompose(DiscreteDomainType const &global_domain, MPICommType comm,
          std::array<int, DiscreteDomainType::rank()> const &par_vector) {
  static_assert(ddc::is_discrete_domain_v<DiscreteDomainType>,
                "DiscreteDomainType must be a DDC discrete domain type");

#ifndef PALIWA_WITH_MPI
  assert(std::all_of(par_vector.begin(), par_vector.end(),
                      [](int i) { return i == 1; }));
  (void)comm;
  process_group::set_state(MPI_COMM_NULL);
  return global_domain;
#else
  constexpr std::size_t dimensionality = DiscreteDomainType::rank();
  using DVect = typename DiscreteDomainType::discrete_vector_type;

#ifndef NDEBUG
  {
    int n = std::reduce(par_vector.begin(), par_vector.end(), 1,
                         std::multiplies<int>());
    int comm_size = -1;
    MPI_Comm_size(comm, &comm_size);
    assert(n == comm_size);
  }
#endif

  std::array<int, dimensionality> periods;
  periods.fill(1); // Periodic in every dimension.

  MPI_Comm cart_comm = MPI_COMM_NULL;
  MPI_Cart_create(comm, static_cast<int>(dimensionality), par_vector.data(),
                   periods.data(), /*reorder=*/true, &cart_comm);

  // set_state builds the full topology cache from cart_comm.
  process_group::set_state(cart_comm);

  // Use the already-cached coords to compute the local subdomain — no
  // further MPI calls needed.
  DVect par_vector_dv;
  std::ranges::transform(par_vector,
                          ddc::detail::array(par_vector_dv).begin(),
                          [](int i) { return static_cast<long int>(i); });
  DVect my_coords_dv;
  for (std::size_t d = 0; d < dimensionality; ++d)
    ddc::detail::array(my_coords_dv)[d] =
        static_cast<long int>(process_group::my_coord(static_cast<int>(d)));

  return distribute_idx_range(global_domain, par_vector_dv, my_coords_dv);
#endif
}

} // namespace paliwa