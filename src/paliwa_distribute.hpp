#pragma once

#include <numeric>
#include <vector>

#include <Kokkos_Core.hpp>
#include <ddc/ddc.hpp>

#ifdef PALIWA_WITH_MPI
#include <mpi.h>
#endif // PALIWA_WITH_MPI

#include "paliwa_transform.hpp"
#include "paliwa_utils.hpp"

namespace paliwa {

struct MPIOptionalGuard {
  MPIOptionalGuard(int &argc, char **&argv) {
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
static ddc::DiscreteDomain<HeadTag, Tags...>
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

#ifdef PALIWA_WITH_MPI
template <typename DiscreteDomainType>
std::pair<DiscreteDomainType, MPI_Comm> decompose_domain_on_communicator(
    DiscreteDomainType const &global_domain, MPI_Comm comm,
    std::array<int, DiscreteDomainType::rank()> const &par_vector) {
  static_assert(ddc::is_discrete_domain_v<DiscreteDomainType>,
                "DiscreteDomainType must be a DDC discrete domain type");
  using DVect = typename DiscreteDomainType::discrete_vector_type;
#ifndef NDEBUG
  constexpr size_t dimensionality = DiscreteDomainType::rank();
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
}


#endif // PALIWA_WITH_MPI
} // namespace paliwa