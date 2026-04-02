// SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include <map>
#include <set>
#include <vector>

#include <ddc/ddc.hpp>

#ifdef PALIWA_WITH_MPI
#include <mpi.h>
#endif // PALIWA_WITH_MPI

#include "paliwa_distribute.hpp"
#include "paliwa_domains.hpp"
#include "paliwa_transform.hpp"

namespace paliwa {

#ifdef PALIWA_WITH_MPI

namespace detail {

/// Info for a single peer in the "ghost" exchange.
template <typename DElem1d> struct PeerExchange {
  int remote_rank;
  std::vector<DElem1d> indices_we_need;
  std::vector<DElem1d> indices_they_need;
};

template <typename DimToTransform, bool IsHierarchization,
          typename LocalDomainType, typename LevelVectorType>
std::vector<PeerExchange<ddc::DiscreteElement<DimToTransform>>>
compute_peer_exchanges(
    std::map<int, std::vector<ddc::DiscreteElement<DimToTransform>>>
        ghost_by_rank,
    ddc::StridedDiscreteDomain<DimToTransform> const &full_1d_strided,
    ddc::DiscreteDomain<DimToTransform> const &global_domain_1d,
    LocalDomainType const &local_domain_1d, LevelVectorType const &level,
    LevelVectorType const &minimum_level, LevelVectorType const &maximum_level,
    std::string const &wavelet_name, MPI_Comm cart_comm, int dim_index) {
  using DElem1d = ddc::DiscreteElement<DimToTransform>;

  int n_dims_cart = 0;
  MPI_Cartdim_get(cart_comm, &n_dims_cart);
  std::vector<int> cart_dims(n_dims_cart), cart_periods(n_dims_cart),
      cart_coords(n_dims_cart);
  MPI_Cart_get(cart_comm, n_dims_cart, cart_dims.data(), cart_periods.data(),
               cart_coords.data());
  int n_ranks_along_dim = cart_dims[dim_index];

  int my_rank = -1;
  MPI_Comm_rank(cart_comm, &my_rank);
  int my_coord = cart_coords[dim_index];

  // For each rank along this dimension, compute what they need from us
  std::map<int, std::vector<DElem1d>> they_need_from_us;
  for (int coord = 0; coord < n_ranks_along_dim; ++coord) {
    if (coord == my_coord)
      continue;
    std::vector<int> remote_coords_v(cart_coords);
    remote_coords_v[dim_index] = coord;
    int remote_rank = -1;
    MPI_Cart_rank(cart_comm, remote_coords_v.data(), &remote_rank);

    auto remote_local_1d = get_rank_local_domain_along_dim<DimToTransform>(
        global_domain_1d, n_ranks_along_dim, coord);
    auto remote_restricted_1d =
        restrict_strided_with_discrete(full_1d_strided, remote_local_1d);
    auto remote_ghost_1d = get_required_transform_domain<DimToTransform>(
        IsHierarchization, full_1d_strided, remote_restricted_1d,
        ddc::select<DimToTransform>(level),
        ddc::select<DimToTransform>(minimum_level),
        ddc::select<DimToTransform>(maximum_level), wavelet_name);

    std::vector<DElem1d> needed;
    ddc::host_for_each(remote_ghost_1d, [&](DElem1d elem) {
      if (local_domain_1d.contains(elem)) {
        needed.push_back(elem);
      }
    });
    if (!needed.empty()) {
      they_need_from_us[remote_rank] = std::move(needed);
    }
  }

  // Merge: for each rank that appears in ghost_by_rank OR
  // they_need_from_us, create a peer entry
  std::set<int> all_peer_ranks;
  for (auto &[rank, _] : ghost_by_rank)
    all_peer_ranks.insert(rank);
  for (auto &[rank, _] : they_need_from_us)
    all_peer_ranks.insert(rank);

  std::vector<PeerExchange<DElem1d>> peers;
  for (int remote_rank : all_peer_ranks) {
    PeerExchange<DElem1d> peer;
    peer.remote_rank = remote_rank;
    if (ghost_by_rank.count(remote_rank)) {
      peer.indices_we_need = std::move(ghost_by_rank[remote_rank]);
    }
    if (they_need_from_us.count(remote_rank)) {
      peer.indices_they_need = std::move(they_need_from_us[remote_rank]);
    }
    peers.push_back(std::move(peer));
  }

  return peers;
}

/**
 * @brief Exchange ghost hyperplane slices via MPI derived datatypes.
 *
 * For each peer, builds hindexed MPI types that describe the memory
 * locations of ghost elements directly in the source/destination spans.
 * One MPI_Sendrecv per peer, no intermediate buffers.
 *
 * @param peers Peer exchange info (indices each side needs)
 * @param local_grid Source data (send from here)
 * @param extended_span Destination data (receive into here)
 * @param local_restricted Local multi-D strided domain (for iterating slices)
 * @param local_restricted_1d Local 1D strided domain along transform dim
 * @param cart_comm Cartesian communicator
 */
template <typename DimToTransform, typename SrcSpanType, typename DstSpanType,
          typename... DDims>
void exchange_ghost_slices(
    std::vector<PeerExchange<ddc::DiscreteElement<DimToTransform>>> const
        &peers,
    SrcSpanType const local_grid, DstSpanType const extended_span,
    ddc::StridedDiscreteDomain<DDims...> const &local_restricted,
    ddc::StridedDiscreteDomain<DimToTransform> const &local_restricted_1d,
    MPI_Comm cart_comm) {
  using value_type = typename SrcSpanType::element_type;
  using DElem1d = ddc::DiscreteElement<DimToTransform>;

  MPIValueType<value_type> mpi_value_type;

  auto byte_offset = [](auto const &span, ddc::DiscreteElement<DDims...> elem) {
    return static_cast<MPI_Aint>(
        (&span(elem) - span.data_handle()) *
        static_cast<std::ptrdiff_t>(sizeof(value_type)));
  };

  // Collect "pole bases": one representative element per (d-1)-D slice
  auto first_local_1d_elem = local_restricted_1d.front();
  std::vector<ddc::DiscreteElement<DDims...>> pole_bases;
  ddc::host_for_each(
      local_restricted, [&](ddc::DiscreteElement<DDims...> elem) {
        if (ddc::select<DimToTransform>(elem) == first_local_1d_elem) {
          pole_bases.push_back(elem);
        }
      });

  // Build displacements for a set of 1D indices × all pole bases
  auto build_displacements = [&](auto const &span,
                                 std::vector<DElem1d> const &indices_1d) {
    std::vector<MPI_Aint> displacements;
    displacements.reserve(indices_1d.size() * pole_bases.size());
    for (auto &d_elem : indices_1d) {
      for (auto &pole_base : pole_bases) {
        displacements.push_back(byte_offset(
            span, paliwa::replace_dim<DimToTransform>(pole_base, d_elem)));
      }
    }
    return displacements;
  };

  // Post all sends and receives non-blocking, then wait.
  // This avoids deadlocks when ranks have different peer orderings.
  std::vector<MPI_Request> requests;
  std::vector<MPI_Datatype> types_to_free;
  requests.reserve(peers.size() * 2);
  types_to_free.reserve(peers.size() * 2);

  for (auto &peer : peers) {
    // Receive type
    auto recv_displacements =
        build_displacements(extended_span, peer.indices_we_need);
    int recv_count = static_cast<int>(recv_displacements.size());
    std::vector<int> recv_blocklens(recv_count, 1);
    MPI_Datatype recv_type;
    MPI_Type_create_hindexed(recv_count, recv_blocklens.data(),
                             recv_displacements.data(), mpi_value_type,
                             &recv_type);
    MPI_Type_commit(&recv_type);
    types_to_free.push_back(recv_type);

    MPI_Request req;
    MPI_Irecv(extended_span.data_handle(), 1, recv_type, peer.remote_rank, 20,
              cart_comm, &req);
    requests.push_back(req);
  }

  for (auto &peer : peers) {
    // Send type
    auto send_displacements =
        build_displacements(local_grid, peer.indices_they_need);
    int send_count = static_cast<int>(send_displacements.size());
    std::vector<int> send_blocklens(send_count, 1);
    MPI_Datatype send_type;
    MPI_Type_create_hindexed(send_count, send_blocklens.data(),
                             send_displacements.data(), mpi_value_type,
                             &send_type);
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

  for (auto &t : types_to_free) {
    MPI_Type_free(&t);
  }
}

/**
 * @brief Core distributed transform for a single dimension.
 *
 * Allocates a single multi-D chunk that is sparse along DimToTransform
 * (local ∪ ghost indices) and covers the local strided domain in all
 * other dimensions. The transform runs once on the full multi-D extended chunk.
 */
template <typename DimToTransform, bool IsHierarchization,
          typename ChunkSpanType, typename ExecSpace, typename... DDims>
bool distributed_transform_in(
    ChunkSpanType const local_grid,
    ddc::StridedDiscreteDomain<DDims...> const &full_strided_domain,
    ddc::DiscreteVector<DDims...> const &level,
    ddc::DiscreteVector<DDims...> const &minimum_level,
    ddc::DiscreteVector<DDims...> const &maximum_level,
    std::string const &wavelet_name, MPI_Comm cart_comm, int dim_index,
    ExecSpace instance) {
  using value_type = typename ChunkSpanType::element_type;
  using DElem1d = ddc::DiscreteElement<DimToTransform>;

  auto local_domain = local_grid.domain();
  auto full_1d_strided = ddc::select<DimToTransform>(full_strided_domain);

  auto local_restricted_1d = restrict_strided_with_discrete(
      full_1d_strided, ddc::select<DimToTransform>(local_domain));

  // Compute ghost zone (1D along transform dimension only)
  auto ghost_domain_1d = get_required_transform_domain<DimToTransform>(
      IsHierarchization, full_1d_strided, local_restricted_1d,
      ddc::select<DimToTransform>(level),
      ddc::select<DimToTransform>(minimum_level),
      ddc::select<DimToTransform>(maximum_level), wavelet_name);

  if (ghost_domain_1d.size() == 0) {
    if constexpr (IsHierarchization) {
      return hierarchize_in<DimToTransform>(local_grid, level, minimum_level,
                                            maximum_level, wavelet_name,
                                            instance);
    } else {
      return dehierarchize_in<DimToTransform>(local_grid, level, minimum_level,
                                              maximum_level, wavelet_name,
                                              instance);
    }
  }

  // Extended 1D domain: local ∪ ghost
  auto extended_1d = union_of_sparse_domains(
      sparse_from_strided_domain(local_restricted_1d), ghost_domain_1d);

  // Multi-D extended domain: sparse along DimToTransform, strided elsewhere
  ddc::SparseDiscreteDomain<DDims...> extended_domain(
      [&]() -> ddc::SparseDiscreteDomain<DDims> {
        if constexpr (std::is_same_v<DDims, DimToTransform>) {
          return extended_1d;
        } else {
          return sparse_from_strided_domain(restrict_strided_with_discrete(
              ddc::select<DDims>(full_strided_domain),
              ddc::select<DDims>(local_domain)));
        }
      }()...);

  ddc::Chunk extended_chunk("extended_buffer", extended_domain,
                            ddc::HostAllocator<value_type>());
  auto extended_span = extended_chunk.span_view();

  // Copy local data into extended chunk
  auto local_restricted =
      restrict_strided_with_discrete(full_strided_domain, local_domain);
  ddc::host_for_each(local_restricted,
                     [&](ddc::DiscreteElement<DDims...> elem) {
                       extended_span(elem) = local_grid(elem);
                     });

  // Compute peer exchanges (what we need / what they need — no MPI)
  auto global_size_1d =
      full_1d_strided.extents().template get<DimToTransform>() *
      full_1d_strided.strides().template get<DimToTransform>();
  ddc::DiscreteDomain<DimToTransform> global_domain_1d(
      DElem1d(full_1d_strided.front()),
      ddc::DiscreteVector<DimToTransform>(global_size_1d));

  auto peers = compute_peer_exchanges<DimToTransform, IsHierarchization>(
      classify_ghost_by_rank<DimToTransform>(ghost_domain_1d, global_domain_1d,
                                             cart_comm, dim_index),
      full_1d_strided, global_domain_1d,
      ddc::select<DimToTransform>(local_domain), level, minimum_level,
      maximum_level, wavelet_name, cart_comm, dim_index);

  // Exchange ghost hyperplane slices
  exchange_ghost_slices<DimToTransform>(peers, local_grid, extended_span,
                                        local_restricted, local_restricted_1d,
                                        cart_comm);

  // Run transform on the multi-D extended chunk
  if constexpr (IsHierarchization) {
    hierarchize_in<DimToTransform>(extended_span, level, minimum_level,
                                   maximum_level, wavelet_name, instance);
  } else {
    dehierarchize_in<DimToTransform>(extended_span, level, minimum_level,
                                     maximum_level, wavelet_name, instance);
  }

  // Copy local results back
  ddc::host_for_each(local_restricted,
                     [&](ddc::DiscreteElement<DDims...> elem) {
                       local_grid(elem) = extended_span(elem);
                     });

  return true;
}

template <typename ChunkSpanType, typename ExecSpace, typename... DDims,
          size_t... Is>
void distributed_hierarchize_impl(
    ChunkSpanType const local_grid,
    ddc::StridedDiscreteDomain<DDims...> const &full_strided_domain,
    ddc::DiscreteVector<DDims...> const &level,
    ddc::DiscreteVector<DDims...> const &minimum_level,
    ddc::DiscreteVector<DDims...> const &maximum_level,
    std::string const &wavelet_name, MPI_Comm cart_comm, ExecSpace instance,
    std::index_sequence<Is...>) {
  [[maybe_unused]] bool unused =
      (distributed_transform_in<DDims, true>(
           local_grid, full_strided_domain, level, minimum_level, maximum_level,
           wavelet_name, cart_comm, static_cast<int>(Is), instance) &&
       ...);
}

template <typename ChunkSpanType, typename ExecSpace, typename... DDims,
          size_t... Is>
void distributed_dehierarchize_impl(
    ChunkSpanType const local_grid,
    ddc::StridedDiscreteDomain<DDims...> const &full_strided_domain,
    ddc::DiscreteVector<DDims...> const &level,
    ddc::DiscreteVector<DDims...> const &minimum_level,
    ddc::DiscreteVector<DDims...> const &maximum_level,
    std::string const &wavelet_name, MPI_Comm cart_comm, ExecSpace instance,
    std::index_sequence<Is...>) {
  [[maybe_unused]] bool unused =
      (distributed_transform_in<DDims, false>(
           local_grid, full_strided_domain, level, minimum_level, maximum_level,
           wavelet_name, cart_comm, static_cast<int>(Is), instance) &&
       ...);
}

} // namespace detail

/**
 * @brief Distributed hierarchization across all dimensions.
 *
 * Processes each dimension sequentially, exchanging ghost data before each
 * dimension's transform (since previous dimensions modify the data).
 * Per dimension, allocates a single multi-D chunk (sparse along the
 * transform dimension, strided along others) and runs the transform once.
 */
template <typename ChunkSpanType,
          typename ExecSpace = Kokkos::DefaultHostExecutionSpace,
          typename... DDims>
void distributed_hierarchize(
    ChunkSpanType const local_grid,
    ddc::StridedDiscreteDomain<DDims...> const &full_strided_domain,
    ddc::DiscreteVector<DDims...> const &level,
    ddc::DiscreteVector<DDims...> const &minimum_level,
    ddc::DiscreteVector<DDims...> const &maximum_level,
    std::string const &wavelet_name, MPI_Comm cart_comm,
    ExecSpace instance = ExecSpace()) {
  detail::distributed_hierarchize_impl(
      local_grid, full_strided_domain, level, minimum_level, maximum_level,
      wavelet_name, cart_comm, instance, std::index_sequence_for<DDims...>{});
}

/**
 * @brief Distributed dehierarchization across all dimensions.
 */
template <typename ChunkSpanType,
          typename ExecSpace = Kokkos::DefaultHostExecutionSpace,
          typename... DDims>
void distributed_dehierarchize(
    ChunkSpanType const local_grid,
    ddc::StridedDiscreteDomain<DDims...> const &full_strided_domain,
    ddc::DiscreteVector<DDims...> const &level,
    ddc::DiscreteVector<DDims...> const &minimum_level,
    ddc::DiscreteVector<DDims...> const &maximum_level,
    std::string const &wavelet_name, MPI_Comm cart_comm,
    ExecSpace instance = ExecSpace()) {
  detail::distributed_dehierarchize_impl(
      local_grid, full_strided_domain, level, minimum_level, maximum_level,
      wavelet_name, cart_comm, instance, std::index_sequence_for<DDims...>{});
}

#endif // PALIWA_WITH_MPI

} // namespace paliwa
