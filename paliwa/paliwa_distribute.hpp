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
  MPIOptionalGuard([[maybe_unused]] int &argc, [[maybe_unused]] char **&argv) {
#ifdef PALIWA_WITH_MPI
    MPI_Init(&argc, &argv);
#endif
  }
  ~MPIOptionalGuard() {
#ifdef PALIWA_WITH_MPI
    process_group_release_at_exit();
    MPI_Finalize();
#endif
  }
  MPIOptionalGuard(MPIOptionalGuard const &) = delete;
  MPIOptionalGuard &operator=(MPIOptionalGuard const &) = delete;

private:
#ifdef PALIWA_WITH_MPI
  // Forward-declared free function, defined after process_group below,
  // so the destructor here doesn't need process_group's full definition
  // at this point in the file.
  static void process_group_release_at_exit();
#endif
};

// ============================================================
// process_group
// ============================================================
// Stores the Cartesian communicator together with cached topology
// information (dimensions, coordinates, and coordinate-to-rank lookup).
//
// Public, version-stable entry points (safe for external code to
// depend on across paliwa releases):
//
//   process_group::init_topology(comm, par_vector)
//       Sets up (or replaces) the Cartesian communicator and topology
//       cache. This is the only function in this file that makes
//       topology-altering MPI calls (MPI_Dims_create, MPI_Cart_create).
//       Call this once per distinct process-grid shape; it is cheap
//       to call again (any previous communicator is freed first), but
//       there's no reason to call it more than once for a fixed
//       par_vector.
//
//   process_group::localize(global_domain)
//       Pure index arithmetic, no MPI calls. Computes the rank-local
//       subdomain for global_domain against the currently cached
//       topology. Safe and cheap to call repeatedly, including once
//       per sparse-grid component domain sharing the same process
//       group with different global extents.
//
//   decompose(global_domain, comm, par_vector)
//       Convenience wrapper: init_topology(comm, par_vector) followed
//       by localize(global_domain). Kept for existing single-domain
//       call sites and tests; equivalent to calling the two steps
//       separately.
//
// Everything else (ghost classification, peer exchange, slice
// exchange) is an internal implementation detail of the current
// nearest-neighbour ghost-exchange model and is expected to change,
// likely to be replaced outright, when the MPI pencil-transpose path
// lands. Do not treat those signatures as stable.

struct process_group {
  process_group() = delete; // Purely static interface.

  // ----------------------------------------------------------
  // Public, stable accessors
  // ----------------------------------------------------------

  static MPICommType get_cart_comm() { return get_state().m_cart_comm; }

  /// Free any communicator currently held. Safe to call multiple
  /// times (idempotent) and safe to call when nothing is held.
  /// Must be called before MPI_Finalize if init_topology() was ever
  /// called; MPIOptionalGuard's destructor does this automatically
  /// when paliwa owns MPI_Init/MPI_Finalize itself.
  static void release() {
    State &s = get_state();
#ifdef PALIWA_WITH_MPI
    if (s.m_cart_comm != MPI_COMM_NULL) {
      MPI_Comm_free(&s.m_cart_comm);
    }
    s.m_n_dims = 0;
    s.m_my_rank = -1;
    s.m_max_ranks_per_dim = 0;
    s.m_cart_dims.clear();
    s.m_my_coords.clear();
    s.m_coord_to_rank.clear();
#endif
    s.m_cart_comm = MPI_COMM_NULL;
  }

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
    return s
        .m_coord_to_rank[static_cast<std::size_t>(d) *
                             static_cast<std::size_t>(s.m_max_ranks_per_dim) +
                         static_cast<std::size_t>(coord)];
  }
#endif // PALIWA_WITH_MPI

  // ----------------------------------------------------------
  // Public, stable entry points
  // ----------------------------------------------------------

  /**
   * @brief Establish (or replace) the Cartesian process-grid topology.
   *
   * par_vector entries follow MPI_Dims_create semantics: a nonzero
   * entry pins the number of ranks along that dimension; a zero entry
   * (the default, i.e. par_vector = {}) is filled in automatically by
   * MPI to balance the remaining ranks across the unconstrained
   * dimensions.
   *
   * IMPORTANT: when paliwa is used as a component backend inside a
   * larger solver that already owns the process-grid decomposition
   * (e.g. a combination-technique driver assigning process groups per
   * component grid), the caller MUST pass the par_vector matching that
   * externally-imposed decomposition. Leaving par_vector as the
   * default {0,...} lets this function invent its own balanced
   * topology, which has no knowledge of and may conflict with a
   * decomposition already fixed elsewhere. The zero-default exists for
   * standalone use (tests, or paliwa driving its own communicator
   * directly), not for embedded use under an external solver.
   *
   * Without MPI, par_vector must be all-zero or all-one; the stored
   * communicator becomes null and every subsequent localize() call
   * returns the global domain unchanged.
   *
   * Any communicator previously held by process_group is freed before
   * the new one is created, so calling this repeatedly (e.g. across
   * gtest cases in one binary) does not leak. The very last
   * communicator created still needs an explicit release() (or
   * reliance on MPIOptionalGuard's destructor) before MPI_Finalize.
   */
  template <std::size_t Rank>
  static void init_topology(MPICommType comm,
                            std::array<int, Rank> par_vector = {}) {
#ifndef PALIWA_WITH_MPI
    assert(std::all_of(par_vector.begin(), par_vector.end(),
                       [](int i) { return i == 0 || i == 1; }));
    (void)comm;
    set_state(MPI_COMM_NULL);
#else
    int comm_size = -1;
    MPI_Comm_size(comm, &comm_size);

    // Fills in any zero entries, balancing ranks across them; leaves
    // nonzero (pinned) entries untouched. A no-op, other than MPI
    // validating the product against comm_size, when par_vector is
    // already fully specified.
    MPI_Dims_create(comm_size, static_cast<int>(Rank), par_vector.data());

#ifndef NDEBUG
    {
      int n = std::reduce(par_vector.begin(), par_vector.end(), 1,
                          std::multiplies<int>());
      assert(n == comm_size);
    }
#endif

    std::array<int, Rank> periods;
    periods.fill(1); // Periodic in every dimension.

    MPI_Comm cart_comm = MPI_COMM_NULL;
    MPI_Cart_create(comm, static_cast<int>(Rank), par_vector.data(),
                    periods.data(), /*reorder=*/true, &cart_comm);

    set_state(cart_comm);
#endif // PALIWA_WITH_MPI
  }

  /**
   * @brief Compute the rank-local subdomain of global_domain against
   *        the currently cached topology.
   *
   * Pure index arithmetic — makes no MPI calls. Safe and cheap to call
   * repeatedly, including once per sparse-grid component domain that
   * shares the same process group but has a different global extent;
   * there is no need to cache or memoize the result on the caller's
   * side.
   *
   * init_topology() must have been called first (directly, or via
   * decompose()); its cached rank/coordinate table drives this
   * function's arithmetic, but this function itself performs no
   * communication.
   */
  template <typename DiscreteDomainType>
  static DiscreteDomainType localize(DiscreteDomainType const &global_domain) {
    static_assert(ddc::is_discrete_domain_v<DiscreteDomainType>,
                  "DiscreteDomainType must be a DDC discrete domain type");
    constexpr std::size_t dimensionality = DiscreteDomainType::rank();
    using DVect = typename DiscreteDomainType::discrete_vector_type;

    State const &s = get_state();

#ifndef PALIWA_WITH_MPI
    (void)s;
    return global_domain;
#else
    if (s.m_cart_comm == MPI_COMM_NULL) {
      return global_domain;
    }

    assert(s.m_n_dims == static_cast<int>(dimensionality));

    DVect par_vector_dv;
    DVect my_coords_dv;
    for (std::size_t d = 0; d < dimensionality; ++d) {
      ddc::detail::array(par_vector_dv)[d] =
          static_cast<long int>(s.m_cart_dims[d]);
      ddc::detail::array(my_coords_dv)[d] =
          static_cast<long int>(s.m_my_coords[d]);
    }

    return distribute_idx_range(global_domain, par_vector_dv, my_coords_dv);
#endif
  }

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

#ifdef PALIWA_WITH_MPI
    if (s.m_cart_comm != MPI_COMM_NULL) {
      MPI_Comm_free(&s.m_cart_comm);
    }
#endif

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
                              static_cast<std::size_t>(s.m_max_ranks_per_dim) +
                          static_cast<std::size_t>(coord)] = rank;
      }
    }
#endif // PALIWA_WITH_MPI
  }
};

#ifdef PALIWA_WITH_MPI
inline void MPIOptionalGuard::process_group_release_at_exit() {
  process_group::release();
}
#endif

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

// ============================================================
// Section — internal: index-range distribution
// ============================================================
// Pure arithmetic, no MPI calls. Used by process_group::localize().

template <class HeadTag, class... Tags>
constexpr ddc::DiscreteDomain<HeadTag, Tags...>
distribute_idx_range(ddc::DiscreteDomain<HeadTag, Tags...> global_idx_range,
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
        local_1d, distribute_idx_range(ddc::select<Tags...>(global_idx_range),
                                       ddc::select<Tags...>(par_vector),
                                       ddc::select<Tags...>(my_coords)));
  }
}

// ============================================================
// Section — internal: nearest-neighbour ghost exchange
// ============================================================
// Everything below is an implementation detail of the current
// single-hop ghost-exchange model, valid as long as filter width
// keeps ghost data on the immediately neighbouring rank. NOT part of
// the stable external interface — expect this section to be replaced
// by the MPI pencil-transpose path.

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
std::map<int, std::vector<ddc::DiscreteElement<Dim>>>
classify_ghost_by_coord(ddc::SparseDiscreteDomain<Dim> const &ghost_domain,
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
std::map<int, std::vector<ddc::DiscreteElement<Dim>>>
classify_ghost_by_rank(ddc::SparseDiscreteDomain<Dim> const &ghost_domain,
                       ddc::DiscreteDomain<Dim> const &global_domain_1d,
                       int dim_index) {
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
    result[process_group::coord_to_rank(dim_index, coord)] = std::move(indices);
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
  auto byte_offset = [](auto const &span, ddc::DiscreteElement<DDims...> elem) {
    return static_cast<MPI_Aint>(
        (&span(elem) - span.data_handle()) *
        static_cast<std::ptrdiff_t>(sizeof(value_type)));
  };

  // Collect one representative multi-D element per (d-1)-D hyperplane
  // slice: all elements whose DimToTransform coordinate matches the
  // first index of the local 1-D domain.
  auto first_local_1d_elem = local_restricted_1d.front();
  std::vector<ddc::DiscreteElement<DDims...>> pole_bases;
  ddc::host_for_each(
      local_restricted, [&](ddc::DiscreteElement<DDims...> elem) {
        if (ddc::select<DimToTransform>(elem) == first_local_1d_elem)
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
    auto send_displ = build_displacements(local_grid, peer.indices_they_need);
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
// Section — public, stable: decompose (convenience wrapper)
// ============================================================

/**
 * @brief Decompose a global domain across MPI ranks: establishes (or
 *        replaces) the process-grid topology and returns this rank's
 *        local subdomain.
 *
 *   auto local_domain = paliwa::decompose(global, MPI_COMM_WORLD);
 *   // or pin specific dimensions, leave the rest automatic:
 *   auto local_domain = paliwa::decompose(global, MPI_COMM_WORLD, {4, 0});
 *   // or fully explicit:
 *   auto local_domain = paliwa::decompose(global, MPI_COMM_WORLD, {2, 2});
 *
 * Equivalent to, and implemented directly as:
 *
 *   process_group::init_topology(comm, par_vector);
 *   return process_group::localize(global_domain);
 *
 * For repeated use against multiple global domains sharing one
 * process group (e.g. sparse-grid component domains), prefer calling
 * init_topology() once and localize() per domain directly, rather
 * than calling decompose() again for each — decompose() re-runs
 * init_topology() (and therefore MPI_Cart_create) every time.
 *
 * par_vector and the embedding-solver caveat about leaving it as the
 * default are documented on process_group::init_topology().
 */
template <typename DiscreteDomainType>
DiscreteDomainType
decompose(DiscreteDomainType const &global_domain, MPICommType comm,
          std::array<int, DiscreteDomainType::rank()> par_vector = {}) {
  process_group::init_topology(comm, par_vector);
  return process_group::localize(global_domain);
}

} // namespace paliwa