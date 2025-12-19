// SPDX-License-Identifier: MIT
// A lot reused from ddc/tests/strided_discrete_domain.cpp
// for these parts: Copyright (C) The DDC development team,
// see DDC's COPYRIGHT.md file

#include <iostream>
#include <vector>

#ifdef PALIWA_WITH_MPI
#include <mpi.h>
#endif // PALIWA_WITH_MPI

#include <ddc/ddc.hpp>

#include "Kokkos_UnorderedMap.hpp"
#include <Kokkos_Core.hpp>

// #include "paliwa_dimensions.hpp" #todo
#include "paliwa_distribute.hpp"
#include "paliwa_domains.hpp"
#include "paliwa_io.hpp"
#include "paliwa_transform.hpp"
#include "paliwa_utils.hpp"
#include "paliwa_wavelets.hpp"

struct X {};
struct Y {};
struct Z {};
struct Vx {};

struct DDimX : ddc::UniformPointSampling<X> {};
using DElemX = ddc::DiscreteElement<DDimX>;

struct DDimY : ddc::UniformPointSampling<Y> {};
using DElemY = ddc::DiscreteElement<DDimY>;

struct DDimZ : ddc::UniformPointSampling<Z> {};
using DElemZ = ddc::DiscreteElement<DDimZ>;
struct DDimVx : ddc::UniformPointSampling<Vx> {};

template <typename... DDims>
std::vector<ddc::Chunk<double, ddc::StridedDiscreteDomain<DDims...>,
                       ddc::DeviceAllocator<double>>>
initialize_combination_scheme(
    std::vector<ddc::StridedDiscreteDomain<DDims...>> const
        &component_grid_domains,
    std::vector<std::array<long int, sizeof...(DDims)>> const &all_levels,
    std::vector<Kokkos::DefaultExecutionSpace> const &instances) {
  using DElem = ddc::DiscreteElement<DDims...>;

  std::vector<ddc::Chunk<double, ddc::StridedDiscreteDomain<DDims...>,
                         ddc::DeviceAllocator<double>>>
      level_data;

  for (size_t grid_index = 0; grid_index < all_levels.size(); ++grid_index) {
    level_data.emplace_back(ddc::Chunk(
        "strided_grid_" + std::to_string(grid_index),
        component_grid_domains[grid_index], ddc::DeviceAllocator<double>()));
    auto strided_grid = level_data.back().span_view();

    // initialize!
    ddc::parallel_for_each(
        instances[grid_index % instances.size()],
        component_grid_domains[grid_index], KOKKOS_LAMBDA(DElem const ixyz) {
          constexpr size_t dimensionality = sizeof...(DDims);
          double const x =
              ddc::coordinate(ddc::DiscreteElement<DDimX>(ixyz)); // ??
          double const y = ddc::coordinate(ddc::DiscreteElement<DDimY>(ixyz));
          double result;
          if constexpr (dimensionality == 3) {
            double const z = ddc::coordinate(ddc::DiscreteElement<DDimZ>(ixyz));
            result = std::cos(3.0 + (x + y + z));
          } else if constexpr (dimensionality == 2) {
            result = std::cos(3.0 + (x + y));
          }
          strided_grid(ixyz) = result;
        });
  }
  return level_data;
}

template <typename... DDims>
void run_combination_technique(
    std::vector<Kokkos::DefaultExecutionSpace> const &instances,
    MPI_Comm comm) {
  constexpr size_t dimensionality = sizeof...(DDims);
  using DDom = ddc::DiscreteDomain<DDims...>;
  using SDDom = ddc::StridedDiscreteDomain<DDims...>;
  using DElem = SDDom::discrete_element_type;
  using DVect = SDDom::discrete_vector_type;
  std::array<long int, dimensionality> maximum_level;
  if constexpr (dimensionality == 3) {
    maximum_level = {5, 6, 7};
  } else if constexpr (dimensionality == 2) {
    maximum_level = {10, 11};
  } else {
    throw std::runtime_error("Dimensionality not supported");
  }
  std::array<long int, dimensionality> resolution;
  std::transform(maximum_level.begin(), maximum_level.end(), resolution.begin(),
                 [](int ml) { return (1 << ml) + 1; });
  DVect resolution_all(resolution);

  // discrete domain in 3d, for the full grid but not allocated yet
  auto const x_domain_with_periodic_point = ddc::init_discrete_space<DDimX>(
      DDimX::init<DDimX>(ddc::Coordinate<X>(0.0), ddc::Coordinate<X>(1.0),
                         ddc::DiscreteVector<DDimX>(resolution[0])));
  ddc::DiscreteDomain<DDimX> const x_domain =
      x_domain_with_periodic_point.remove_last(ddc::DiscreteVector<DDimX>(1));
  auto const y_domain_with_periodic_point = ddc::init_discrete_space<DDimY>(
      DDimY::init<DDimY>(ddc::Coordinate<Y>(0.0), ddc::Coordinate<Y>(1.0),
                         ddc::DiscreteVector<DDimY>(resolution[1])));
  ddc::DiscreteDomain<DDimY> const y_domain =
      y_domain_with_periodic_point.remove_last(ddc::DiscreteVector<DDimY>(1));
  DDom dom_all;
  if constexpr (dimensionality == 3) {
    auto const z_domain_with_periodic_point = ddc::init_discrete_space<DDimZ>(
        DDimZ::init<DDimZ>(ddc::Coordinate<Z>(0.0), ddc::Coordinate<Z>(1.0),
                           ddc::DiscreteVector<DDimZ>(resolution[2])));
    ddc::DiscreteDomain<DDimZ> const z_domain =
        z_domain_with_periodic_point.remove_last(ddc::DiscreteVector<DDimZ>(1));

    dom_all = DDom(x_domain, y_domain, z_domain);
  } else if constexpr (dimensionality == 2) {
    dom_all = DDom(x_domain, y_domain);
  }

  std::array<int, dimensionality> parallelization_vector = {2, 2};
  auto const [local_domain, cartesian_comm] =
      paliwa::decompose_domain_on_communicator(dom_all, comm,
                                               parallelization_vector);

  std::array<long int, dimensionality> minimum_level;
  std::vector<std::array<long int, dimensionality>> all_levels;
  std::vector<double> all_combi_coefficients;
  if constexpr (dimensionality == 3) {
    minimum_level = {4, 5, 6};
    all_levels = {{4, 6, 7}, {5, 5, 7}, {5, 6, 6}, {4, 5, 6}};
    all_combi_coefficients = {1, 1, 1, -2};
  } else if constexpr (dimensionality == 2) {
    minimum_level = {2, 3};
    all_levels = {{2, 11}, {3, 10}, {4, 9},  {5, 8},  {6, 7}, {7, 6},
                  {8, 5},  {9, 4},  {10, 3}, {2, 10}, {3, 9}, {4, 8},
                  {5, 7},  {6, 6},  {7, 5},  {8, 4},  {9, 3}};
    all_combi_coefficients = {1,  1,  1,  1,  1,  1,  1,  1, 1,
                              -1, -1, -1, -1, -1, -1, -1, -1};
  }
  DVect const ddc_minimum_level(minimum_level);
  DVect const ddc_maximum_level(maximum_level);
  assert(all_levels.size() == all_combi_coefficients.size());
  auto component_grid_domains =
      paliwa::get_strided_domains<DDims...>(all_levels, maximum_level);
  std::vector<ddc::Chunk<double, SDDom, ddc::DeviceAllocator<double>>>
      level_data = initialize_combination_scheme(component_grid_domains,
                                                 all_levels, instances);
  paliwa::fence_all_instances(instances);

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
    paliwa::dump_chunk_span_to_binary_file(strided_grid, filename);
    // std::cout << strided_grid << std::endl; (-> issue)
    ddc::print_content(std::cout, strided_grid) << std::endl;
  }

  std::string const wavelet_name = "biorthogonal";
  // hierarchize / wavelet-ify / filter in each direction
  for (size_t grid_index = 0; grid_index < all_levels.size(); ++grid_index) {
    DVect level(all_levels[grid_index]);
    SDDom const &strided_domain = component_grid_domains[grid_index];
    auto strided_grid = level_data[grid_index].span_view();

    paliwa::hierarchize(strided_grid, level, ddc_minimum_level,
                        ddc_maximum_level, wavelet_name,
                        instances[grid_index % instances.size()]);
  }
  paliwa::fence_all_instances(instances);

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
    paliwa::iterate_hierarchical_subspaces(level, tmp_level, 0,
                                           insert_function);
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
          paliwa::strided_hierarchical_domain_from_level<DDims...>(
              subspace_level, maximum_level);
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
  paliwa::fence_all_instances(instances);

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
  paliwa::fence_all_instances(instances);

  //   de-hierarchize on the combined full grid
  paliwa::dehierarchize(full_grid_view, ddc_maximum_level, ddc_minimum_level,
                        ddc_maximum_level, wavelet_name, instances[0]);
  paliwa::fence_all_instances(instances);

  std::string max_level_str = "";
  for (auto l : maximum_level) {
    max_level_str += std::to_string(l) + "_";
  }
  max_level_str += std::to_string(dimensionality) + "d";
  std::string const filename = "full_grid_" + max_level_str + ".raw";
  paliwa::dump_chunk_span_to_binary_file(full_grid_view, filename);
  std::cout << "Wrote full grid to " << filename << std::endl;
  double const mean_value =
      ddc::parallel_transform_reduce(dom_all, 0., ddc::reducer::sum<double>(),
                                     full_grid_view) /
      dom_all.size();
  std::cout << "Mean value on finest grid: " << mean_value << std::endl;
  if (std::abs(mean_value - (-0.650446)) > 1e-6) {
    throw std::runtime_error(
        "Error: mean value does not match expected value!");
  }
}

template <size_t dimensionality>
void run_combination_technique_in_dimensions(
    std::vector<Kokkos::DefaultExecutionSpace> const &instances,
    MPI_Comm comm) {
  if constexpr (dimensionality == 2) {
    run_combination_technique<DDimX, DDimY>(instances, comm);
  } else if constexpr (dimensionality == 3) {
    run_combination_technique<DDimX, DDimY, DDimZ>(instances, comm);
  } else if constexpr (dimensionality == 4) {
    run_combination_technique<DDimX, DDimY, DDimZ, DDimVx>(instances, comm);
  } else {
    throw std::runtime_error("Dimensionality not yet supported");
  }
}

int main(int argc, char **argv) {
  [[maybe_unused]] paliwa::MPIOptionalGuard mpi(argc, argv);
  Kokkos::ScopeGuard const kokkos_scope(argc, argv);
#ifndef NDEBUG
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  for (int i = 0; i < world_size; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (i == world_rank) {
      std::cout << "paliwa rank " << world_rank << " : device_id "
                << Kokkos::device_id() << std::endl;
      // Kokkos::print_configuration(std::cout);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
#endif // not NDEBUG
  ddc::ScopeGuard const ddc_scope;

  // use up to 32 concurrent streams
  auto instances = Kokkos::Experimental::partition_space(
      Kokkos::DefaultExecutionSpace(), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

  run_combination_technique_in_dimensions<2>(instances, MPI_COMM_WORLD);
}