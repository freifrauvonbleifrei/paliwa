// SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include <Kokkos_Core.hpp>
#include <ddc/ddc.hpp>
#include <numeric>
#include <vector>

#ifdef PALIWA_WITH_MPI
#include <mpi.h>
#endif // PALIWA_WITH_MPI

#include "paliwa_transform.hpp"
#include "paliwa_utils.hpp"

namespace paliwa {

#ifdef PALIWA_WITH_MPI
using MPICommType = MPI_Comm;
#else
using MPICommType = void *; // dummy type when MPI is not enabled
#define MPI_COMM_WORLD nullptr
#define MPI_COMM_NULL nullptr
#endif // PALIWA_WITH_MPI

struct MPIOptionalGuard {
  MPIOptionalGuard([[maybe_unused]] int &argc, [[maybe_unused]] char **&argv) {
#ifdef PALIWA_WITH_MPI
    MPI_Init(&argc, &argv);
#endif // PALIWA_WITH_MPI
  }
  ~MPIOptionalGuard() {
#ifdef PALIWA_WITH_MPI
    MPI_Finalize();
#endif // PALIWA_WITH_MPI
  }
};

template <class HeadTag, class... Tags>
constexpr ddc::DiscreteDomain<HeadTag, Tags...>
distribute_idx_range(ddc::DiscreteDomain<HeadTag, Tags...> global_idx_range,
                     ddc::DiscreteVector<HeadTag, Tags...> const &par_vector,
                     ddc::DiscreteVector<HeadTag, Tags...> const &my_coords) {
  ddc::DiscreteDomain<HeadTag> global_idx_range_along_dim =
      ddc::select<HeadTag>(global_idx_range);
  ddc::DiscreteDomain<HeadTag> local_idx_range_along_dim;
  ddc::DiscreteDomain<Tags...> remaining_idx_range;

  // The number of MPI processes along this dimension
  auto n_ranks_along_dim = ddc::DiscreteVector<HeadTag>(par_vector);
  auto rank_along_dim = ddc::DiscreteVector<HeadTag>(my_coords);
  // Calculate the local index range
  if (global_idx_range_along_dim.size() % n_ranks_along_dim != 0) {
    throw std::runtime_error(
        "The provided index range cannot be split equally over "
        "the specified number of MPI ranks.");
  }

  ddc::DiscreteVector<HeadTag> elems_on_dim(global_idx_range_along_dim.size() /
                                            n_ranks_along_dim);
  ddc::DiscreteElement<HeadTag> distrib_start(
      global_idx_range_along_dim.front() + rank_along_dim * elems_on_dim);
  local_idx_range_along_dim =
      ddc::DiscreteDomain<HeadTag>(distrib_start, elems_on_dim);
  if constexpr (sizeof...(Tags) > 0) {
    // Calculate the index range for the subsequent dimensions
    ddc::DiscreteDomain<Tags...> remaining_dims =
        ddc::select<Tags...>(global_idx_range);
    remaining_idx_range =
        distribute_idx_range(remaining_dims, ddc::select<Tags...>(par_vector),
                             ddc::select<Tags...>(my_coords));
  }
  return ddc::DiscreteDomain<HeadTag, Tags...>(local_idx_range_along_dim,
                                               remaining_idx_range);
}

template <typename DiscreteDomainType>
constexpr std::pair<DiscreteDomainType, paliwa::MPICommType>
decompose_domain_on_communicator(
    DiscreteDomainType const &global_domain, paliwa::MPICommType comm,
    std::array<int, DiscreteDomainType::rank()> const &par_vector) {
  static_assert(ddc::is_discrete_domain_v<DiscreteDomainType>,
                "DiscreteDomainType must be a DDC discrete domain type");
  using DVect = typename DiscreteDomainType::discrete_vector_type;
#ifndef PALIWA_WITH_MPI
  assert(std::all_of(par_vector.begin(), par_vector.end(), [](int i) {
    return i == 1;
  })); // no parallelization without MPI
  return {global_domain, MPI_COMM_NULL};
#else // PALIWA_WITH_MPI
  constexpr size_t dimensionality = DiscreteDomainType::rank();
#ifndef NDEBUG
  assert(dimensionality == par_vector.size());
  auto num_procs = std::reduce(std::begin(par_vector), std::end(par_vector), 1,
                               std::multiplies<int>());
  int comm_size = -1;
  MPI_Comm_size(comm, &comm_size);
  assert(num_procs == comm_size);
#endif // NDEBUG
  std::array<int, dimensionality> periods;
  for (size_t i = 0; i < dimensionality; ++i) {
    periods[i] = 1;
  }
  int reorder = true;
  MPI_Comm comm_cart = MPI_COMM_NULL;

  MPI_Cart_create(comm, static_cast<int>(dimensionality), par_vector.data(),
                  periods.data(), reorder, &comm_cart);
  int my_rank = -1;
  MPI_Comm_rank(comm_cart, &my_rank);
  std::array<int, dimensionality> my_coords;
  MPI_Cart_coords(comm_cart, my_rank, static_cast<int>(dimensionality),
                  my_coords.data());

  DVect par_vector_dv;
  std::ranges::transform(par_vector, ddc::detail::array(par_vector_dv).begin(),
                         [](int i) { return static_cast<long int>(i); });
  DVect my_coords_dv;
  std::ranges::transform(my_coords, ddc::detail::array(my_coords_dv).begin(),
                         [](int i) { return static_cast<long int>(i); });
  // dimension-recursive call
  return {distribute_idx_range(global_domain, par_vector_dv, my_coords_dv),
          comm_cart};
#endif // PALIWA_WITH_MPI
}

/**
 * @brief Get the 1D local domain owned by a given Cartesian coordinate along a
 * dimension. Assumes uniform partition of the global domain.
 */
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

/**
 * @brief Classify ghost indices by owner coordinate along a dimension.
 *
 * @return Map from owner coordinate (0-based) to list of ghost indices.
 */
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
    if (owner_coord >= n_ranks_along_dim) {
      owner_coord = n_ranks_along_dim - 1;
    }
    result[owner_coord].push_back(elem);
  });

  return result;
}

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

/**
 * @brief Classify ghost indices by owner MPI rank along a Cartesian dimension.
 *
 * Thin wrapper around classify_ghost_by_coord that converts Cartesian
 * coordinates to MPI ranks.
 */
template <typename Dim>
std::map<int, std::vector<ddc::DiscreteElement<Dim>>>
classify_ghost_by_rank(ddc::SparseDiscreteDomain<Dim> const &ghost_domain,
                       ddc::DiscreteDomain<Dim> const &global_domain_1d,
                       MPI_Comm cart_comm, int dim_index) {
  int n_dims = 0;
  MPI_Cartdim_get(cart_comm, &n_dims);
  std::vector<int> dims(n_dims), periods(n_dims), my_coords(n_dims);
  MPI_Cart_get(cart_comm, n_dims, dims.data(), periods.data(),
               my_coords.data());

  auto by_coord = classify_ghost_by_coord<Dim>(ghost_domain, global_domain_1d,
                                               dims[dim_index]);

  std::map<int, std::vector<ddc::DiscreteElement<Dim>>> result;
  for (auto &[coord, indices] : by_coord) {
    std::vector<int> target_coords(my_coords);
    target_coords[dim_index] = coord;
    int target_rank = -1;
    MPI_Cart_rank(cart_comm, target_coords.data(), &target_rank);
    result[target_rank] = std::move(indices);
  }
  return result;
}

#endif // PALIWA_WITH_MPI

} // namespace paliwa
