// SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include <array>
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

template <std::size_t N>
constexpr std::array<int, N> identity_cartesian_axes() {
  std::array<int, N> axes{};
  for (std::size_t d = 0; d < N; ++d) {
    axes[d] = static_cast<int>(d);
  }
  return axes;
}

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
 * Describe each regular (d-1)-D slice with nested vector types, then
 * select the requested slices with one displacement per axis index.
 * Source and destination may have different memory strides.
 * Nonblocking sends and receives, followed by a wait before transforming.
 *
 * @param peers Peer exchange info (indices each side needs)
 * @param local_grid Source data (send from here)
 * @param receive_grid Destination data (receive into here)
 * @param local_restricted Local multi-D strided domain (for iterating slices)
 * @param cart_comm Cartesian communicator
 */
template <typename DimToTransform, typename SrcSpanType, typename DstSpanType,
          typename... DDims>
void exchange_ghost_slices(
    std::vector<PeerExchange<ddc::DiscreteElement<DimToTransform>>> const
        &peers,
    SrcSpanType const local_grid, DstSpanType const receive_grid,
    ddc::StridedDiscreteDomain<DDims...> const &local_restricted,
    MPI_Comm cart_comm) {
  using value_type = typename SrcSpanType::element_type;
  using DElem1d = ddc::DiscreteElement<DimToTransform>;

  MPIValueType<value_type> mpi_value_type;

  auto byte_offset = [](auto const &span, ddc::DiscreteElement<DDims...> elem) {
    return static_cast<MPI_Aint>(
        (&span(elem) - span.data_handle()) *
        static_cast<std::ptrdiff_t>(sizeof(value_type)));
  };

  // The non-transformed axes are regular in both the local strided span
  // and the sparse receive span (which stores those same axes densely).
  // Use actual memory strides so the two layouts need not match.
  auto build_slice = [&](auto const &span, DElem1d first) {
    auto const anchor = replace_dim(local_restricted.front(), first);
    auto axis_stride = [&]<typename Dim>() -> MPI_Aint {
      if constexpr (std::is_same_v<Dim, DimToTransform>) {
        return 0;
      } else {
        auto const axis = ddc::select<Dim>(local_restricted);
        if (axis.size() < 2)
          return 0;
        auto const next = axis.front() + axis.strides();
        return byte_offset(span, replace_dim(anchor, next)) -
               byte_offset(span, anchor);
      }
    };
    std::array<MPI_Aint, sizeof...(DDims)> const strides {
      axis_stride.template operator()<DDims>()...
    };
    std::array<int, sizeof...(DDims)> const counts{
        (std::is_same_v<DDims, DimToTransform>
             ? 1
             : static_cast<int>(
                   ddc::select<DDims>(local_restricted).size()))...};
    MPI_Datatype slice;
    MPI_Type_contiguous(1, mpi_value_type, &slice);
    // Last dimension varies fastest, matching DDC's iteration order.
    for (size_t d = counts.size(); d-- > 0;) {
      if (counts[d] == 1)
        continue;
      MPI_Datatype outer;
      MPI_Type_create_hvector(counts[d], 1, strides[d], slice, &outer);
      MPI_Type_free(&slice);
      slice = outer;
    }
    return slice;
  };

  std::vector<MPI_Request> requests;
  std::vector<MPI_Datatype> types_to_free;
  requests.reserve(peers.size() * 2);
  types_to_free.reserve(peers.size() * 2 + 2);
  MPI_Datatype send_slice = MPI_DATATYPE_NULL;
  MPI_Datatype receive_slice = MPI_DATATYPE_NULL;
  auto build_type = [&](auto const &span, std::vector<DElem1d> const &indices,
                        MPI_Datatype &slice) {
    if (slice == MPI_DATATYPE_NULL) {
      slice = build_slice(span, indices.front());
      types_to_free.push_back(slice);
    }
    std::vector<MPI_Aint> displacements;
    displacements.reserve(indices.size());
    for (auto index : indices)
      displacements.push_back(
          byte_offset(span, replace_dim(local_restricted.front(), index)));
    MPI_Datatype type;
    // MUST 1.11.2 does not track MPI_Type_create_hindexed_block.
    std::vector<int> const block_lengths(indices.size(), 1);
    MPI_Type_create_hindexed(static_cast<int>(indices.size()),
                             block_lengths.data(), displacements.data(), slice,
                             &type);
    MPI_Type_commit(&type);
    types_to_free.push_back(type);
    return type;
  };

  // Post all sends and receives before waiting, including one-way peers.
  for (auto &peer : peers) {
    if (peer.indices_we_need.empty())
      continue;
    auto const recv_type =
        build_type(receive_grid, peer.indices_we_need, receive_slice);

    MPI_Request req;
    MPI_Irecv(receive_grid.data_handle(), 1, recv_type, peer.remote_rank, 20,
              cart_comm, &req);
    requests.push_back(req);
  }

  for (auto &peer : peers) {
    if (peer.indices_they_need.empty())
      continue;
    auto const send_type =
        build_type(local_grid, peer.indices_they_need, send_slice);

    MPI_Request req;
    MPI_Isend(local_grid.data_handle(), 1, send_type, peer.remote_rank, 20,
              cart_comm, &req);
    requests.push_back(req);
  }

  MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
              MPI_STATUSES_IGNORE);

  for (auto &t : types_to_free) {
    MPI_Type_free(&t);
  }
}

/**
 * @brief Core distributed transform for a single dimension.
 *
 * Packed poles gather directly from local storage and a remote-only sparse
 * buffer. Direct and pass-wise execution retain a combined local/ghost buffer.
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
    ExecSpace instance, PoleExecution pole_execution) {
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

  auto const local_restricted =
      restrict_strided_with_discrete(full_strided_domain, local_domain);
  auto const increasing = std::views::iota(
      static_cast<long>(ddc::select<DimToTransform>(minimum_level)) + 1,
      static_cast<long>(ddc::select<DimToTransform>(level)) + 1);
  auto transform = [&](auto grid, auto remote) {
    if constexpr (IsHierarchization) {
      transform_with_remote<DimToTransform>(
          grid, remote, level, maximum_level, increasing | std::views::reverse,
          lifting_wavelet_filter_offsets_and_coefficients.at(wavelet_name),
          local_restricted, instance, pole_execution);
    } else {
      transform_with_remote<DimToTransform>(
          grid, remote, level, maximum_level, increasing,
          lifting_wavelet_reconstruct_offsets_and_coefficients.at(wavelet_name),
          local_restricted, instance, pole_execution);
    }
    instance.fence();
  };

  // No incoming ghosts does not imply no communication: another rank may
  // still need our local coefficients (e.g. the coarse point of a tiny slab).
  // Even without peers, restrict writes to the local poles in other dimensions.
  if (peers.empty()) {
    transform(local_grid, local_grid);
    return true;
  }

  if constexpr (Kokkos::SpaceAccessibility<ExecSpace,
                                           Kokkos::HostSpace>::accessible) {
    if (pole_execution == PoleExecution::PackedPoles) {
      // Only received values need sparse storage. Local values stay in the
      // strided grid and are gathered/scattered directly by each packed pole.
      ddc::SparseDiscreteDomain<DDims...> remote_domain(
          [&]() -> ddc::SparseDiscreteDomain<DDims> {
            if constexpr (std::is_same_v<DDims, DimToTransform>)
              return ghost_domain_1d;
            else
              return sparse_from_strided_domain(
                  ddc::select<DDims>(local_restricted));
          }()...);
      ddc::Chunk remote_chunk("remote_buffer", remote_domain,
                              ddc::HostAllocator<value_type>());
      auto const remote = remote_chunk.span_view();
      exchange_ghost_slices<DimToTransform>(peers, local_grid, remote,
                                            local_restricted, cart_comm);
      transform(local_grid, remote);
      return true;
    }
  }

  // Extended 1D domain: local ∪ ghost
  auto extended_1d = union_of_sparse_domains(
      sparse_from_strided_domain(local_restricted_1d), ghost_domain_1d);

  // Combined sparse domain, with local strided coordinates on other axes
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
  ddc::host_for_each(local_restricted,
                     [&](ddc::DiscreteElement<DDims...> elem) {
                       extended_span(elem) = local_grid(elem);
                     });

  // Exchange ghost hyperplane slices
  exchange_ghost_slices<DimToTransform>(peers, local_grid, extended_span,
                                        local_restricted, cart_comm);

  transform(extended_span, extended_span);

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
    std::array<int, sizeof...(DDims)> const &cartesian_axes,
    std::index_sequence<Is...>, PoleExecution pole_execution) {
  [[maybe_unused]] bool unused =
      (distributed_transform_in<DDims, true>(
           local_grid, full_strided_domain, level, minimum_level, maximum_level,
           wavelet_name, cart_comm, cartesian_axes[Is], instance,
           pole_execution) &&
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
    std::array<int, sizeof...(DDims)> const &cartesian_axes,
    std::index_sequence<Is...>, PoleExecution pole_execution) {
  [[maybe_unused]] bool unused =
      (distributed_transform_in<DDims, false>(
           local_grid, full_strided_domain, level, minimum_level, maximum_level,
           wavelet_name, cart_comm, cartesian_axes[Is], instance,
           pole_execution) &&
       ...);
}

} // namespace detail

/**
 * @brief Distributed hierarchization across all dimensions.
 * @param cartesian_axes Permutation mapping each domain dimension to an axis
 * of cart_comm. Defaults to the domain dimension order.
 * @param pole_execution Optional CPU pole packing and loop order; devices use
 * direct execution. Defaults to packed poles on the CPU.
 *
 * Processes each dimension sequentially, exchanging ghost data before each
 * dimension's transform (since previous dimensions modify the data).
 * Packed poles use a sparse receive buffer containing only remote values;
 * local values stay in their original strided storage.
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
    ExecSpace instance = ExecSpace(),
    std::array<int, sizeof...(DDims)> const &cartesian_axes =
        detail::identity_cartesian_axes<sizeof...(DDims)>(),
    PoleExecution pole_execution = PoleExecution::PackedPoles) {
  detail::distributed_hierarchize_impl(
      local_grid, full_strided_domain, level, minimum_level, maximum_level,
      wavelet_name, cart_comm, instance, cartesian_axes,
      std::index_sequence_for<DDims...>{}, pole_execution);
}

/**
 * @brief Distributed dehierarchization across all dimensions.
 * @param cartesian_axes Permutation mapping each domain dimension to an axis
 * of cart_comm. Defaults to the domain dimension order.
 * @param pole_execution Optional CPU pole packing and loop order; devices use
 * direct execution. Defaults to packed poles on the CPU.
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
    ExecSpace instance = ExecSpace(),
    std::array<int, sizeof...(DDims)> const &cartesian_axes =
        detail::identity_cartesian_axes<sizeof...(DDims)>(),
    PoleExecution pole_execution = PoleExecution::PackedPoles) {
  detail::distributed_dehierarchize_impl(
      local_grid, full_strided_domain, level, minimum_level, maximum_level,
      wavelet_name, cart_comm, instance, cartesian_axes,
      std::index_sequence_for<DDims...>{}, pole_execution);
}

#endif // PALIWA_WITH_MPI

} // namespace paliwa
