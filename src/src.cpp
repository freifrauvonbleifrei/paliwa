// SPDX-License-Identifier: MIT
// A lot reused from ddc/tests/strided_discrete_domain.cpp
// for these parts: Copyright (C) The DDC development team,
// see DDC's COPYRIGHT.md file

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <ranges>
#include <vector>

#include <ddc/ddc.hpp>
#include <mpi.h>

#include "Kokkos_UnorderedMap.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#define PERIODIC_DOMAIN // Comment this to run non-periodic simulation

#define DIMENSIONALITY 2
static constexpr int8_t dimensionality = DIMENSIONALITY;

struct X {
#if defined(PERIODIC_DOMAIN)
  static constexpr bool PERIODIC = true;
#else
  static constexpr bool PERIODIC = false;
  static_assert(!PERIODIC, "Non-periodic case not implemented/untested");
#endif
};
struct Y {
#if defined(PERIODIC_DOMAIN)
  static constexpr bool PERIODIC = true;
#else
  static constexpr bool PERIODIC = false;
#endif
};
#if DIMENSIONALITY > 2
struct Z {
#if defined(PERIODIC_DOMAIN)
  static constexpr bool PERIODIC = true;
#else
  static constexpr bool PERIODIC = false;
#endif
};
#endif

struct DDimX : ddc::UniformPointSampling<X> {};
using DElemX = ddc::DiscreteElement<DDimX>;

struct DDimY : ddc::UniformPointSampling<Y> {};
using DElemY = ddc::DiscreteElement<DDimY>;

#if DIMENSIONALITY > 2
struct DDimZ : ddc::UniformPointSampling<Z> {};
using DElemZ = ddc::DiscreteElement<DDimZ>;
#endif

using DElemXY = ddc::DiscreteElement<DDimX, DDimY>;
using DVectXY = ddc::DiscreteVector<DDimX, DDimY>;
using DDomXY = ddc::DiscreteDomain<DDimX, DDimY>;
using SDDomXY = ddc::StridedDiscreteDomain<DDimX, DDimY>;

#if DIMENSIONALITY > 2
using DElemXYZ = ddc::DiscreteElement<DDimX, DDimY, DDimZ>;
using DVectXYZ = ddc::DiscreteVector<DDimX, DDimY, DDimZ>;
using DDomXYZ = ddc::DiscreteDomain<DDimX, DDimY, DDimZ>;
using SDDomXYZ = ddc::StridedDiscreteDomain<DDimX, DDimY, DDimZ>;

using DElem = DElemXYZ;
using DVect = DVectXYZ;
using DDom = DDomXYZ;
using SDDom = SDDomXYZ;

// DElemZ constexpr lbound_z = ddc::init_trivial_half_bounded_space<DDimZ>();
// DElem constexpr lbound_all(lbound_x, lbound_y, lbound_z);
DElem constexpr lbound_all(0, 0, 0);

#else // DIMENSIONALITY > 2

using DElem = DElemXY;
using DVect = DVectXY;
using DDom = DDomXY;
using SDDom = SDDomXY;

// DElem constexpr lbound_all(lbound_x, lbound_y);
DElem constexpr lbound_all(0, 0);

#endif // DIMENSIONALITY > 2

double const x_start = 0.;
double const x_end = 1.;
double const y_start = 0.;
double const y_end = 1.;
double const z_start = 0.;
double const z_end = 1.;

SDDom strided_domain_from_level(
    std::array<long int, dimensionality> const &level,
    std::array<long int, dimensionality> const &finest_level) {
  std::array<long int, dimensionality> resolution;
  std::ranges::transform(level, resolution.begin(),
                         [](long int l) { return (1 << l); });
  DVect resolution_all;
  ddc::detail::array(resolution_all) =
      resolution; // TODO temporary solution until assignment from std::array is
                  // implemented
  std::array<long int, dimensionality> level_diff;
  std::ranges::transform(level, finest_level, level_diff.begin(),
                         [](long int l, long int ml) { return ml - l; });
  std::array<long int, dimensionality> stride;
  std::ranges::transform(level_diff, stride.begin(),
                         [](long int l) { return (1 << l); });
  DVect strides_all;
  ddc::detail::array(strides_all) = stride; // TODO
  return SDDom(lbound_all, resolution_all, strides_all);
}

SDDom strided_hierarchical_domain_from_level(
    std::array<long int, dimensionality> const &level,
    std::array<long int, dimensionality> const &finest_level) {
  std::array<long int, dimensionality> resolution;
  std::ranges::transform(level, resolution.begin(),
                         [](long int ml) { return (1 << (ml - 1)); });
  std::array<long int, dimensionality> level_diff;
  std::ranges::transform(level, finest_level, level_diff.begin(),
                         [](long int l, long int ml) { return ml - l; });
  std::array<long int, dimensionality> half_stride;
  std::ranges::transform(level_diff, half_stride.begin(),
                         [](long int l) { return (1 << l); });
  // special case level 0
  for (size_t i = 0; i < dimensionality; ++i) {
    assert(level[i] <= finest_level[i]);
    assert(level[i] >= 0);
    if (level[i] == 0) {
      resolution[i] = 1;  // the coarsest level has one point
      half_stride[i] = 0; // no offset
    }
  }
  DVect resolution_all;
  ddc::detail::array(resolution_all) =
      resolution; // TODO temporary solution until assignment from std::array is
                  // implemented
  DVect half_stride_vect;
  ddc::detail::array(half_stride_vect) = half_stride; // TODO
  DElem start_all = lbound_all + half_stride_vect;
  std::array<long int, dimensionality> stride;
  std::ranges::transform(level_diff, stride.begin(),
                         [](long int l) { return (1 << (l + 1)); });
  DVect strides_all;
  ddc::detail::array(strides_all) = stride; // TODO
  std::cout << " subs " << start_all << " " << resolution_all << " "
            << strides_all << std::endl;
  return SDDom(start_all, resolution_all, strides_all);
}

template <typename DDimInWhichItsOdd>
SDDom odd_strided_domain_from_domain(SDDom const &domain) {
  ddc::DiscreteVector<DDimInWhichItsOdd> odd_offset(
      domain.strides().get<DDimInWhichItsOdd>());
  DVect strides_odd = domain.strides();
  strides_odd.get<DDimInWhichItsOdd>() *= 2;
  auto extent_odd = domain.extents();
  extent_odd.get<DDimInWhichItsOdd>() =
      (extent_odd.get<DDimInWhichItsOdd>()) / 2;
  return SDDom(lbound_all + odd_offset, extent_odd, strides_odd);
}

template <typename DDimInWhichItsEven>
SDDom even_strided_domain_from_domain(SDDom const &domain) {
  DVect strides_even = domain.strides();
  strides_even.get<DDimInWhichItsEven>() *= 2;
  auto extent_even = domain.extents();
  extent_even.get<DDimInWhichItsEven>() =
      (extent_even.get<DDimInWhichItsEven>()) / 2;
  return SDDom(lbound_all, extent_even, strides_even);
}

// TODO consider making this constexpr frozen::map ?
static const std::map<std::string,
                      std::vector<std::pair<int, std::array<double, 3>>>>
    lifting_wavelet_filter_offsets_and_coefficients = {
        {"hat", {{1, {-0.5, 1.0, -0.5}}}},
        {"biorthogonal", {{1, {-0.5, 1.0, -0.5}}, {0, {0.25, 1.0, 0.25}}}},
        {"fullweighting", {{0, {0.25, 0.5, 0.25}}, {1, {-0.5, 1.0, -0.5}}}},
};
static const std::map<std::string,
                      std::vector<std::pair<int, std::array<double, 3>>>>
    lifting_wavelet_reconstruct_offsets_and_coefficients = {
        {"hat", {{1, {0.5, 1.0, 0.5}}}},
        {"biorthogonal", {{0, {-0.25, 1.0, 0.25}}, {1, {0.5, 1.0, 0.5}}}},
        {"fullweighting", {{1, {0.5, 1.0, 0.5}}, {0, {-0.5, 2.0, -0.5}}}},
};

template <typename InWhichDim>
ddc::DiscreteVector<InWhichDim>
get_dimension_component(std::array<long int, dimensionality> const &level) {
  DVect ddc_level;
  ddc::detail::array(ddc_level) = level; // TODO temporary solution until
                                         // assignment from std::array is
                                         // implemented
  return ddc::DiscreteVector<InWhichDim>(ddc_level);
}

template <typename DDimInWhichToHierarchize,
          typename DDomainType,   // TODO either DDom or SDDom
          typename ChunkSpanType, // TODO w.r.t. DDomainType
          typename LevelRange,    // TODO input_range concept
          typename ExecSpace = Kokkos::DefaultExecutionSpace>
void transform_in(DDomainType const &strided_domain,
                  ChunkSpanType const strided_grid,
                  std::array<long int, dimensionality> const &level,
                  std::array<long int, dimensionality> const &maximum_level,
                  LevelRange const &one_d_level_range,
                  std::vector<std::pair<int, std::array<double, 3>>> const
                      &lifting_offsets_and_coefficients,
                  ExecSpace instance = ExecSpace()) {
  auto const ddc_level_1d_vec =
      get_dimension_component<DDimInWhichToHierarchize>(level);
  auto const ddc_max_level_1d_vec =
      get_dimension_component<DDimInWhichToHierarchize>(maximum_level);

  // check the finest stride, if DDomainType is SDDom
  if constexpr (std::is_same_v<DDomainType, SDDom>) {
    assert((1 << (ddc_max_level_1d_vec - ddc_level_1d_vec)) ==
           strided_domain.strides().template get<DDimInWhichToHierarchize>());
  }

  DVect current_level;
  ddc::detail::array(current_level) = level; // TODO temporary solution until
                                             // assignment from std::array is
                                             // implemented

  for (long int current_1d_level : one_d_level_range) {
    int const current_stride = (1 << (ddc_max_level_1d_vec - current_1d_level));
    current_level.get<DDimInWhichToHierarchize>() = current_1d_level;
    auto const operating_domain = strided_domain_from_level(
        ddc::detail::array(current_level), maximum_level);

    for (auto const &[offset, filter] : lifting_offsets_and_coefficients) {
      std::function<SDDom(SDDom const &)> coarsen_domain;
      if (offset == 0) {
        coarsen_domain =
            even_strided_domain_from_domain<DDimInWhichToHierarchize>;
      } else if (offset == 1) {
        coarsen_domain =
            odd_strided_domain_from_domain<DDimInWhichToHierarchize>;
      } else {
        throw std::runtime_error("Filter offset not supported");
      }
      auto const write_to_domain = coarsen_domain(operating_domain);

      ddc::parallel_for_each(
          instance, write_to_domain, KOKKOS_LAMBDA(DElem const ixyz) {
            // how to access / slice at every other point in x?
            // check for out of bounds
            if (((offset == 1) &&
                 (ddc::DiscreteElement<DDimInWhichToHierarchize>(ixyz) +
                      ddc::DiscreteVector<DDimInWhichToHierarchize>(
                          current_stride) <=
                  ddc::DiscreteElement<DDimInWhichToHierarchize>(
                      write_to_domain.back()))) ||
                ((offset == 0) &&
                 (ddc::DiscreteElement<DDimInWhichToHierarchize>(ixyz) >
                  ddc::DiscreteElement<DDimInWhichToHierarchize>(
                      lbound_all)))) {
              strided_grid(ixyz) =
                  filter[0] *
                      strided_grid(
                          ixyz - ddc::DiscreteVector<DDimInWhichToHierarchize>(
                                     current_stride)) +
                  filter[1] * strided_grid(ixyz) +
                  filter[2] *
                      strided_grid(
                          ixyz + ddc::DiscreteVector<DDimInWhichToHierarchize>(
                                     current_stride));
            } else {
              if (offset == 1) {
                // on the upper boundary, no +1 available
                // TODO make separate step to avoid branch here
                DElem wraparound;
                if constexpr (std::is_same_v<DDimInWhichToHierarchize, DDimX>) {
                  wraparound = DElem(DElemX(lbound_all), DElemY(ixyz));
                } else if constexpr (std::is_same_v<DDimInWhichToHierarchize,
                                                    DDimY>) {
                  wraparound = DElem(DElemX(ixyz), DElemY(lbound_all));
                } else {
                  static_assert("Not implemented for this dimension");
                }
                strided_grid(ixyz) =
                    filter[0] *
                        strided_grid(
                            ixyz -
                            ddc::DiscreteVector<DDimInWhichToHierarchize>(
                                current_stride)) +
                    filter[1] * strided_grid(ixyz) +
                    filter[2] * strided_grid(wraparound);

              } else {
                // on the lower boundary, no -1 available
                auto const domain_back = operating_domain.back();
                DElem wraparound;
                if constexpr (std::is_same_v<DDimInWhichToHierarchize, DDimX>) {
                  wraparound = DElem(DElemX(domain_back), DElemY(ixyz));
                } else if constexpr (std::is_same_v<DDimInWhichToHierarchize,
                                                    DDimY>) {
                  wraparound = DElem(DElemX(ixyz), DElemY(domain_back));
                } else {
                  static_assert("Not implemented for this dimension");
                }
                strided_grid(ixyz) =
                    filter[0] * strided_grid(wraparound) +
                    filter[1] * strided_grid(ixyz) +
                    filter[2] *
                        strided_grid(
                            ixyz +
                            ddc::DiscreteVector<DDimInWhichToHierarchize>(
                                current_stride));
              }
            }
          });
    }
  }
}

template <typename DDimInWhichToHierarchize, typename DDomainType,
          typename ChunkSpanType,
          typename ExecSpace = Kokkos::DefaultExecutionSpace>
void hierarchize_in(DDomainType const &strided_domain,
                    ChunkSpanType const strided_grid,
                    std::array<long int, dimensionality> const &level,
                    std::array<long int, dimensionality> const &minimum_level,
                    std::array<long int, dimensionality> const &maximum_level,
                    std::string const &wavelet_name = "hat",
                    ExecSpace instance = ExecSpace()) {

  auto const ddc_level_1d_vec =
      get_dimension_component<DDimInWhichToHierarchize>(level);
  auto const ddc_min_level_1d_vec =
      get_dimension_component<DDimInWhichToHierarchize>(minimum_level);

  auto decreasing_range =
      std::views::iota(static_cast<long int>(ddc_min_level_1d_vec) - 1,
                       static_cast<long int>(ddc_level_1d_vec) + 1) |
      std::views::reverse;
  return transform_in<DDimInWhichToHierarchize>(
      strided_domain, strided_grid, level, maximum_level, decreasing_range,
      lifting_wavelet_filter_offsets_and_coefficients.at(wavelet_name),
      instance);
}

template <typename DDimInWhichToHierarchize, typename DDomainType,
          typename ChunkSpanType,
          typename ExecSpace = Kokkos::DefaultExecutionSpace>
void dehierarchize_in(DDomainType const &strided_domain,
                      ChunkSpanType const strided_grid,
                      std::array<long int, dimensionality> const &level,
                      std::array<long int, dimensionality> const &minimum_level,
                      std::array<long int, dimensionality> const &maximum_level,
                      std::string const &wavelet_name = "hat",
                      ExecSpace instance = ExecSpace()) {

  auto const ddc_level_1d_vec =
      get_dimension_component<DDimInWhichToHierarchize>(level);
  auto const ddc_min_level_1d_vec =
      get_dimension_component<DDimInWhichToHierarchize>(minimum_level);

  auto increasing_range =
      std::views::iota(static_cast<long int>(ddc_min_level_1d_vec) - 1,
                       static_cast<long int>(ddc_level_1d_vec) + 1);
  return transform_in<DDimInWhichToHierarchize>(
      strided_domain, strided_grid, level, maximum_level, increasing_range,
      lifting_wavelet_reconstruct_offsets_and_coefficients.at(wavelet_name),
      instance);
}

template <typename T> // with T for example std::array<long int, dimensionality>
void iterate_hierarchical_subspaces(
    const T &nodal_level, T &tmp_level, size_t current_dim,
    const std::function<void(const T &)> &callback) {
  assert(tmp_level.size() == nodal_level.size());
  if (current_dim < nodal_level.size()) {
    for (tmp_level[current_dim] = 0;
         tmp_level[current_dim] <= nodal_level[current_dim];
         ++tmp_level[current_dim]) {
      iterate_hierarchical_subspaces(nodal_level, tmp_level, current_dim + 1,
                                     callback);
    }
  } else {
    callback(tmp_level);
  }
}

template <typename ChunkType>
void dump_chunk_span_to_binary_file(ChunkType const span,
                                    std::string const &filename) {
  auto host_mirror_view =
      ddc::create_mirror_view_and_copy(Kokkos::SharedHostPinnedSpace(), span);
  std::ofstream file(filename, std::ios::trunc | std::ios::binary);
  for (size_t i = 0; i < host_mirror_view.size(); ++i) {
    file.write(reinterpret_cast<char *>(&host_mirror_view.data_handle()[i]),
               sizeof(host_mirror_view.data_handle()[i]));
  }
}

template <class HeadTag>
static ddc::DiscreteDomain<HeadTag>
distribute_idx_range(ddc::DiscreteDomain<HeadTag> global_idx_range,
                     DVect const &par_vector, DVect const &my_coords) {
  if (global_idx_range.size() % par_vector.get<HeadTag>() != 0) {
    throw std::runtime_error(
        "The provided index range cannot be split equally over "
        "the specified number of MPI ranks.");
  }
  ddc::DiscreteVector<HeadTag> elems_on_dim(global_idx_range.size() /
                                            par_vector.get<HeadTag>());
  ddc::DiscreteDomain<HeadTag> last_dim_local_idx_range(
      global_idx_range.front() + my_coords.get<HeadTag>() * elems_on_dim,
      elems_on_dim);
  return last_dim_local_idx_range;
}

template <class HeadTag, class... Tags,
          std::enable_if_t<(sizeof...(Tags) > 0), bool> = true>
static ddc::DiscreteDomain<HeadTag, Tags...>
distribute_idx_range(ddc::DiscreteDomain<HeadTag, Tags...> global_idx_range,
                     DVect const &par_vector, DVect const &my_coords) {
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
  // Calculate the index range for the subsequent dimensions
  ddc::DiscreteDomain<Tags...> remaining_dims =
      ddc::select<Tags...>(global_idx_range);
  remaining_idx_range =
      distribute_idx_range(remaining_dims, par_vector, my_coords);
  return ddc::DiscreteDomain<HeadTag, Tags...>(local_idx_range_along_dim,
                                               remaining_idx_range);
}

DDom decompose_domain_on_communicator(
    DDom const &global_domain, MPI_Comm comm,
    std::array<int, dimensionality> const &par_vector) {
  static_assert(ddc::is_discrete_domain_v<DDom>);
#ifndef NDEBUG
  assert(dimensionality == par_vector.size());
  assert(dimensionality == global_domain.rank());
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
  return distribute_idx_range(global_domain, par_vector_dv, my_coords_dv);
}

template <typename InstancesType>
void fence_all_instances(InstancesType const &instances) {
  for (auto const &instance : instances) {
    instance.fence();
  }
}

template <int8_t dimensionality>
void run_combination_technique(
    std::vector<Kokkos::DefaultExecutionSpace> const &instances,
    MPI_Comm comm) {
  // TODO implement
}

int main() {
  MPI_Init(0, nullptr);
  Kokkos::ScopeGuard const kokkos_scope;
  ddc::ScopeGuard const ddc_scope;
  Kokkos::print_configuration(std::cout);

  // use up to 32 concurrent streams
  auto instances = Kokkos::Experimental::partition_space(
      Kokkos::DefaultExecutionSpace(), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

  run_combination_technique<dimensionality>(instances, MPI_COMM_WORLD);
#if DIMENSIONALITY > 2
  std::array<long int, dimensionality> const maximum_level = {5, 6, 7};
#else
  std::array<long int, dimensionality> const maximum_level = {4, 5};
#endif
  std::array<long int, dimensionality> resolution;
  std::transform(maximum_level.begin(), maximum_level.end(), resolution.begin(),
                 [](int ml) { return (1 << ml) + 1; });
  DVect resolution_all;
  ddc::detail::array(resolution_all) =
      resolution; // TODO temporary solution until assignment from std::array is
                  // implemented
  // discrete domain in 3d, for the full grid but not allocated yet
  auto const x_domain_with_periodic_point = ddc::init_discrete_space<DDimX>(
      DDimX::init<DDimX>(ddc::Coordinate<X>(x_start), ddc::Coordinate<X>(x_end),
                         ddc::DiscreteVector<DDimX>(resolution[0])));
  ddc::DiscreteDomain<DDimX> const x_domain =
      x_domain_with_periodic_point.remove_last(ddc::DiscreteVector<DDimX>(1));
  auto const y_domain_with_periodic_point = ddc::init_discrete_space<DDimY>(
      DDimY::init<DDimY>(ddc::Coordinate<Y>(y_start), ddc::Coordinate<Y>(y_end),
                         ddc::DiscreteVector<DDimY>(resolution[1])));
  ddc::DiscreteDomain<DDimY> const y_domain =
      y_domain_with_periodic_point.remove_last(ddc::DiscreteVector<DDimY>(1));
#if DIMENSIONALITY > 2
  auto const z_domain_with_periodic_point = ddc::init_discrete_space<DDimZ>(
      DDimZ::init<DDimZ>(ddc::Coordinate<Z>(z_start), ddc::Coordinate<Z>(z_end),
                         ddc::DiscreteVector<DDimZ>(resolution[2])));
  ddc::DiscreteDomain<DDimZ> const z_domain =
      z_domain_with_periodic_point.remove_last(ddc::DiscreteVector<DDimZ>(1));

  DDom const dom_all(x_domain, y_domain, z_domain);
#else
  DDom const dom_all(x_domain, y_domain);
#endif
  std::array<int, dimensionality> parallelization_vector = {2, 2};
  DDom const local_domain = decompose_domain_on_communicator(
      dom_all, MPI_COMM_WORLD, parallelization_vector);

  for (int i = 0; i < world_size; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (i == world_rank) {
      std::cout << "rank " << world_rank << " : " << std::endl;
      std::cout << " local domain: " << ddc::coordinate(local_domain.front())
                << " " << ddc::coordinate(local_domain.back()) << std::endl;
    }
  }

#if DIMENSIONALITY > 2
  std::array<long int, dimensionality> const minimum_level = {4, 5, 6};
  std::vector<std::array<long int, dimensionality>> all_levels = {
      {4, 6, 7}, {5, 5, 7}, {5, 6, 6}, {4, 5, 6}};
  std::vector<double> all_combi_coefficients = {1, 1, 1, -2};
#else
  std::array<long int, dimensionality> const minimum_level = {2, 3};
  std::vector<std::array<long int, dimensionality>> all_levels = {
      {2, 5}, {3, 4}, {4, 3}, {2, 4}, {3, 3}};
  std::vector<double> all_combi_coefficients = {1, 1, 1, -1, -1};
#endif
  assert(all_levels.size() == all_combi_coefficients.size());
  std::vector<SDDom> component_grid_domains;
  // TODO these as Kokkos unordered_map?
  std::vector<ddc::Chunk<double, SDDom, ddc::DeviceAllocator<double>>>
      level_data;

  for (size_t grid_index = 0; grid_index < all_levels.size(); ++grid_index) {
    auto &level = all_levels[grid_index];
    component_grid_domains.emplace_back(
        strided_domain_from_level(level, maximum_level));
    level_data.emplace_back(ddc::Chunk(
        "strided_grid_" + std::to_string(grid_index),
        component_grid_domains.back(), ddc::DeviceAllocator<double>()));
    auto strided_grid = level_data.back().span_view();

    // initialize!
    ddc::parallel_for_each(
        instances[grid_index % instances.size()], component_grid_domains.back(),
        KOKKOS_LAMBDA(DElem const ixyz) {
          double const x =
              ddc::coordinate(ddc::DiscreteElement<DDimX>(ixyz)); // ??
          double const y = ddc::coordinate(ddc::DiscreteElement<DDimY>(ixyz));
#if DIMENSIONALITY > 2
          double const z = ddc::coordinate(ddc::DiscreteElement<DDimZ>(ixyz));
          strided_grid(ixyz) = std::cos(3.0 + (x + y + z));
#else
          strided_grid(ixyz) = std::cos(3.0 + (x + y));
#endif
        });
  }
  fence_all_instances(instances);

  for (size_t grid_index = 0; grid_index < all_levels.size(); ++grid_index) {
    auto &level = all_levels[grid_index];
    auto strided_grid = level_data[grid_index].span_view();

    // TODO how easiest for visualizable output? pdi? raw ofstream? (-> raw
    // ofstream for now)
    std::string level_str = "";
    for (auto l : level) {
      level_str += std::to_string(l) + "_";
    }
    level_str += std::to_string(dimensionality) + "d";
    std::string const filename = "strided_grid_" + level_str + ".raw";
    dump_chunk_span_to_binary_file(strided_grid, filename);
    // std::cout << strided_grid << std::endl; (-> issue)
    ddc::print_content(std::cout, strided_grid) << std::endl;
  }

  std::string const wavelet_name = "biorthogonal";
  // hierarchize / wavelet-ify / filter in each direction
  for (size_t grid_index = 0; grid_index < all_levels.size(); ++grid_index) {
    // todo this for is another potential parallel_for_each!
    auto &level = all_levels[grid_index];
    SDDom const &strided_domain = component_grid_domains[grid_index];
    auto strided_grid = level_data[grid_index].span_view();

    hierarchize_in<DDimX>(strided_domain, strided_grid, level, minimum_level,
                          maximum_level, wavelet_name,
                          instances[grid_index % instances.size()]);
    hierarchize_in<DDimY>(strided_domain, strided_grid, level, minimum_level,
                          maximum_level, wavelet_name,
                          instances[grid_index % instances.size()]);
#if DIMENSIONALITY > 2
    hierarchize_in<DDimZ>(strided_domain, strided_grid, level, minimum_level,
                          maximum_level, wavelet_name,
                          instances[grid_index % instances.size()]);
#endif
  }
  fence_all_instances(instances);

  // sparse grid!
  std::map<std::array<long int, dimensionality>, int> subspace_count;
  // for each component grid, count up the contained subspaces
  std::function<void(const std::array<long int, dimensionality> &)>
      insert_function =
          [&](const std::array<long int, dimensionality> &subspace_level) {
            if (subspace_count.find(subspace_level) == subspace_count.end()) {
              subspace_count[subspace_level] = 1;
            } else {
              ++subspace_count[subspace_level];
            }
          };
  for (const auto &level : all_levels) {
    std::array<long int, dimensionality> tmp_level;
    iterate_hierarchical_subspaces(level, tmp_level, 0, insert_function);
  }

  Kokkos::UnorderedMap<size_t, std::array<long int, dimensionality>,
                       Kokkos::DefaultExecutionSpace>
      subspaces_levels(subspace_count.size());
  Kokkos::UnorderedMap<size_t, std::pair<SDDom, double *>,
                       Kokkos::DefaultExecutionSpace>
      subspaces_domains_and_data_pointers(subspace_count.size());
  // leads to weird hangup (different type w/ hash function?)!!
  // auto subspaces_levels_host = Kokkos::create_mirror(subspaces_levels);
  Kokkos::UnorderedMap<size_t, std::array<long int, dimensionality>,
                       Kokkos::SharedHostPinnedSpace>
      subspaces_levels_host(subspace_count.size());
  Kokkos::UnorderedMap<size_t, std::pair<SDDom, double *>,
                       Kokkos::SharedHostPinnedSpace>
      subspaces_domains_and_data_pointers_host(subspace_count.size());

  size_t accumulated_size = 0;
  size_t used_subspace_number = 0;
  for (const auto &subspace_level_and_count : subspace_count) {
    auto const &subspace_level = subspace_level_and_count.first;
    auto const &count = subspace_level_and_count.second;
    if (count > 1) {
      auto subspace_domain =
          strided_hierarchical_domain_from_level(subspace_level, maximum_level);
      accumulated_size += subspace_domain.size();
      subspaces_levels_host.insert(used_subspace_number, subspace_level);
      subspaces_domains_and_data_pointers_host.insert(
          used_subspace_number++,
          std::make_pair(std::move(subspace_domain), nullptr));
    }
  }
  std::cout << "Total size of all subspaces: " << accumulated_size << std::endl;

  // allocate once
  Kokkos::View<double *> all_subspace_data("all_subspace_data",
                                           accumulated_size);

  size_t current_data_pointer_index = 0;
  for (size_t i = 0; i < subspaces_domains_and_data_pointers_host.capacity();
       ++i) {
    if (subspaces_domains_and_data_pointers_host.valid_at(i)) {
      auto &subspace_domain =
          subspaces_domains_and_data_pointers_host.value_at(i).first;
      auto &data_pointer =
          subspaces_domains_and_data_pointers_host.value_at(i).second;
      // basically exclusive scan
      data_pointer = all_subspace_data.data() + current_data_pointer_index;
      current_data_pointer_index += subspace_domain.size();
    }
  }
  Kokkos::deep_copy(subspaces_levels, subspaces_levels_host);
  Kokkos::deep_copy(subspaces_domains_and_data_pointers,
                    subspaces_domains_and_data_pointers_host);

  // collect component grids onto the sparse grid
  for (size_t grid_index = 0; grid_index < all_levels.size(); ++grid_index) {
    // todo this for is another potential parallel_for_each!
    auto &level = all_levels[grid_index];
    double coefficient = all_combi_coefficients[grid_index];
    auto strided_grid = level_data[grid_index].span_view();

    for (size_t i = 0; i < subspaces_domains_and_data_pointers_host.capacity();
         ++i) {
      if (subspaces_domains_and_data_pointers_host.valid_at(i)) {
        auto const &subspace_level = subspaces_levels_host.value_at(i);
        auto const &subspace_domain =
            subspaces_domains_and_data_pointers_host.value_at(i).first;
        auto const &data_pointer =
            subspaces_domains_and_data_pointers_host.value_at(i).second;
        bool contains = true; // TODO use ddc contains domain operator
        for (size_t d = 0; d < dimensionality; ++d) {
          if (subspace_level[d] > level[d]) {
            contains = false;
            break;
          }
        }
        if (contains == false)
          continue;
        // copy data into the allocated space
        ddc::ChunkSpan<double, SDDom> subspace_chunk_span(data_pointer,
                                                          subspace_domain);
        auto subspace_view = subspace_chunk_span.span_view();
        ddc::parallel_for_each(
            instances[i % instances.size()], subspace_domain,
            KOKKOS_LAMBDA(DElem const ixyz) {
              subspace_view(ixyz) += coefficient * strided_grid(ixyz);
            });
      }
    }
  }
  fence_all_instances(instances);

  // interpolate all onto full grid
  // allocate and initialize to 0
  ddc::Chunk full_grid("interpolated_on_full_grid", dom_all,
                       ddc::DeviceAllocator<double>());
  auto full_grid_view = full_grid.span_view();

  // now copy into full grid
  for (size_t i = 0; i < subspaces_domains_and_data_pointers_host.capacity();
       ++i) {
    if (subspaces_domains_and_data_pointers_host.valid_at(i)) {
      auto const &subspace_domain =
          subspaces_domains_and_data_pointers_host.value_at(i).first;
      auto const &data_pointer =
          subspaces_domains_and_data_pointers_host.value_at(i).second;
      ddc::ChunkSpan<double, SDDom> subspace_chunk_span(data_pointer,
                                                        subspace_domain);
      auto subspace_view = subspace_chunk_span.span_view();
      ddc::parallel_for_each(
          instances[i % instances.size()], subspace_domain,
          KOKKOS_LAMBDA(DElem const ixyz) {
            full_grid_view(ixyz) += subspace_view(ixyz);
          });
    }
  }
  fence_all_instances(instances);

  //   de-hierarchize on the combined full grid
  dehierarchize_in<DDimX, DDom>(dom_all, full_grid_view, maximum_level,
                                minimum_level, maximum_level, wavelet_name);
  dehierarchize_in<DDimY, DDom>(dom_all, full_grid_view, maximum_level,
                                minimum_level, maximum_level, wavelet_name);

  std::string max_level_str = "";
  for (auto l : maximum_level) {
    max_level_str += std::to_string(l) + "_";
  }
  max_level_str += std::to_string(dimensionality) + "d";
  std::string const filename = "full_grid_" + max_level_str + ".raw";
  dump_chunk_span_to_binary_file(full_grid_view, filename);
  std::cout << "Wrote full grid to " << filename << std::endl;
  double const mean_value =
      ddc::parallel_transform_reduce(dom_all, 0., ddc::reducer::sum<double>(),
                                     full_grid_view) /
      dom_all.size();
  std::cout << "Mean value on finest grid: " << mean_value << std::endl;
  if (std::abs(mean_value - (-0.650446)) > 1e-6) {
    std::cerr << "Error: mean value does not match expected value!"
              << std::endl;
    return 1;
  }
  MPI_Finalize();
}