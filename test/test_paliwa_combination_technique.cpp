// SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-3.0-or-later

// Some parts reused from ddc/tests/strided_discrete_domain.cpp
// for these parts:
// Copyright (C) The DDC development team, see DDC's COPYRIGHT.md file

// SPDX-License-Identifier: MIT

#include <iostream>
#include <vector>

#ifdef PALIWA_WITH_MPI
#include <mpi.h>
#endif // PALIWA_WITH_MPI

#include <gtest/gtest.h>

#include <ddc/ddc.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>

#include "../paliwa/paliwa_dimensions.hpp"
#include "../paliwa/paliwa_distribute.hpp"
#include "../paliwa/paliwa_distributed_transform.hpp"
#include "../paliwa/paliwa_domains.hpp"
#include "../paliwa/paliwa_io.hpp"
#include "../paliwa/paliwa_transform.hpp"
#include "../paliwa/paliwa_utils.hpp"
#include "../paliwa/paliwa_wavelets.hpp"

constexpr double pi = 3.14159265358979323846;
double sinusoid_integral_analytical(int d) {
  // Analytical integral: ∫[0,1]^n sin(π x_1) ... sin(π x_n) dx = (2/π)^n
  return std::pow(2.0 / pi, d);
}

template <typename InstancesType, typename... DDims>
std::vector<ddc::Chunk<double, ddc::StridedDiscreteDomain<DDims...>,
                       ddc::DeviceAllocator<double>>>
initialize_combination_scheme(
    std::vector<ddc::StridedDiscreteDomain<DDims...>> const
        &component_grid_domains,
    std::vector<std::array<long int, sizeof...(DDims)>> const &all_levels,
    InstancesType const &instances) {
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
          auto coordinate = ddc::coordinate(ixyz).array();
          double result = 1.0;
          for (double xi : coordinate) {
            result *= std::sin(pi * xi);
          }
          strided_grid(ixyz) = result;
        });
  }
  return level_data;
}

template <typename... DDims, typename InstancesType>
void run_combination_technique(InstancesType const &instances,
                               paliwa::MPICommType comm) {
  constexpr size_t dimensionality = sizeof...(DDims);
  using SDDom = ddc::StridedDiscreteDomain<DDims...>;
  using DElem = SDDom::discrete_element_type;
  using DVect = SDDom::discrete_vector_type;
  std::array<long int, dimensionality> maximum_level;
  if constexpr (dimensionality == 2) {
    maximum_level = {10, 11};
  } else if constexpr (dimensionality == 3) {
    maximum_level = {5, 6, 7};
  } else if constexpr (dimensionality == 4) {
    maximum_level = {3, 4, 5, 6};
  } else {
    throw std::runtime_error("Dimensionality not supported");
  }
  std::array<long int, dimensionality> resolution;
  std::transform(maximum_level.begin(), maximum_level.end(), resolution.begin(),
                 [](int ml) { return (1 << ml); });
  DVect resolution_all(resolution);

  ddc::DiscreteDomain<DDims...> dom_all =
      paliwa::optional_initialize_dims_periodic_unit_cube<DDims...>(
          resolution_all);

  std::array<long int, dimensionality> minimum_level;
  std::vector<std::array<long int, dimensionality>> all_levels;
  std::vector<double> all_combi_coefficients;
  if constexpr (dimensionality == 2) {
    minimum_level = {2, 3};
    all_levels = {{2, 11}, {3, 10}, {4, 9},  {5, 8},  {6, 7}, {7, 6},
                  {8, 5},  {9, 4},  {10, 3}, {2, 10}, {3, 9}, {4, 8},
                  {5, 7},  {6, 6},  {7, 5},  {8, 4},  {9, 3}};
    all_combi_coefficients = {1,  1,  1,  1,  1,  1,  1,  1, 1,
                              -1, -1, -1, -1, -1, -1, -1, -1};
  } else if constexpr (dimensionality == 3) {
    minimum_level = {4, 5, 6};
    all_levels = {{4, 6, 7}, {5, 5, 7}, {5, 6, 6}, {4, 5, 6}};
    all_combi_coefficients = {1, 1, 1, -2};
  } else if constexpr (dimensionality == 4) {
    minimum_level = {1, 2, 3, 4};
    all_levels = {{2, 2, 4, 4}, {3, 2, 3, 4}, {1, 4, 3, 4}, {1, 2, 5, 4},
                  {2, 3, 3, 4}, {1, 3, 4, 4}, {2, 2, 3, 5}, {1, 3, 3, 5},
                  {1, 2, 3, 6}, {1, 2, 4, 5}, {1, 2, 4, 4}, {2, 2, 3, 4},
                  {1, 3, 3, 4}, {1, 2, 3, 5}, {1, 2, 3, 4}};
    all_combi_coefficients = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -3, -3, -3, -3, 3};
  } else {
    static_assert(dimensionality < 5, "Dimensionality not yet supported");
  }
  DVect const ddc_minimum_level(minimum_level);
  DVect const ddc_maximum_level(maximum_level);
  assert(all_levels.size() == all_combi_coefficients.size());
  auto component_grid_domains =
      paliwa::get_strided_domains<DDims...>(all_levels, maximum_level);
  // TODO make this full-grid compatible class?
  std::vector<ddc::Chunk<double, SDDom, ddc::DeviceAllocator<double>>>
      level_data = initialize_combination_scheme(component_grid_domains,
                                                 all_levels, instances);
  size_t accumulated_full_grid_size = 0;
  for (const auto &domain : component_grid_domains) {
    accumulated_full_grid_size += domain.size();
  }
  if constexpr (dimensionality == 2) {
    EXPECT_EQ(accumulated_full_grid_size, 106496);
  } else if constexpr (dimensionality == 3) {
    EXPECT_EQ(accumulated_full_grid_size, 425984);
  } else if constexpr (dimensionality == 4) {
    EXPECT_EQ(accumulated_full_grid_size, 50176);
  }
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
    // std::cout << strided_grid << std::endl;
  }

  std::string const wavelet_name = "biorthogonal";
  // hierarchize / wavelet-ify / filter in each direction
  for (size_t grid_index = 0; grid_index < all_levels.size(); ++grid_index) {
    DVect level(all_levels[grid_index]);
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

  // TODO make this a sparse-grid compatible class?
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

  size_t accumulated_sparse_grid_size = 0;
  size_t used_subspace_number = 0;
  for (const auto &subspace_level_and_count : subspace_count) {
    auto const &subspace_level = subspace_level_and_count.first;
    auto const &count = subspace_level_and_count.second;
    if (count > 1) {
      auto subspace_domain =
          paliwa::strided_hierarchical_domain_from_level<DDims...>(
              subspace_level, maximum_level);
      accumulated_sparse_grid_size += subspace_domain.size();
      subspaces_levels_host.insert(used_subspace_number, subspace_level);
      subspaces_domains_and_data_pointers_host.insert(
          used_subspace_number++,
          std::make_pair(std::move(subspace_domain), nullptr));
    }
  }
  if constexpr (dimensionality == 2) {
    EXPECT_EQ(accumulated_sparse_grid_size, 18432);
  } else if constexpr (dimensionality == 3) {
    EXPECT_EQ(accumulated_sparse_grid_size, 131072);
  } else if constexpr (dimensionality == 4) {
    EXPECT_EQ(accumulated_sparse_grid_size, 5120);
  }

  // allocate once
  Kokkos::View<double *> all_subspace_data("all_subspace_data",
                                           accumulated_sparse_grid_size);

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
  ddc::parallel_fill(full_grid_view, 0);

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
  double const reduced_value = ddc::parallel_transform_reduce(
      dom_all, static_cast<double>(0.), ddc::reducer::sum<double>(),
      full_grid_view);
  double const mean_value = reduced_value / dom_all.size();
  std::cout << "Mean value on finest grid: " << mean_value << " (from "
            << reduced_value << " total)" << std::endl;
  EXPECT_NEAR(mean_value, sinusoid_integral_analytical(dimensionality), 0.031);
}

template <size_t dimensionality, typename InstancesType>
void run_combination_technique_in_dimensions(InstancesType const &instances,
                                             paliwa::MPICommType comm) {
  if constexpr (dimensionality == 2) {
    run_combination_technique<paliwa::DDimA, paliwa::DDimB>(instances, comm);
  } else if constexpr (dimensionality == 3) {
    run_combination_technique<paliwa::DDimC, paliwa::DDimD, paliwa::DDimE>(
        instances, comm);
  } else if constexpr (dimensionality == 4) {
    run_combination_technique<paliwa::DDimF, paliwa::DDimG, paliwa::DDimH,
                              paliwa::DDimI>(instances, comm);
  } else {
    throw std::runtime_error("Dimensionality not yet supported");
  }
}

TEST(combination_technique, full_integration_2d) {
  // use up to 32 concurrent streams
  auto instances = Kokkos::Experimental::partition_space(
      Kokkos::DefaultExecutionSpace(), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
  run_combination_technique_in_dimensions<2>(instances, MPI_COMM_WORLD);
}

TEST(combination_technique, full_integration_3d) {
  auto instances = Kokkos::Experimental::partition_space(
      Kokkos::DefaultExecutionSpace(), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
  run_combination_technique_in_dimensions<3>(instances, MPI_COMM_WORLD);
}

TEST(combination_technique, full_integration_4d) {
  auto instances = Kokkos::Experimental::partition_space(
      Kokkos::DefaultExecutionSpace(), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
  run_combination_technique_in_dimensions<4>(instances, MPI_COMM_WORLD);
}

#ifdef PALIWA_WITH_MPI

void test_distributed_combination_technique_2d() {
  using DimX = paliwa::DDimR;
  using DimY = paliwa::DDimS;
  using SDDom = ddc::StridedDiscreteDomain<DimX, DimY>;
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

  std::array<long int, 2> maximum_level = {6, 7};
  std::array<long int, 2> minimum_level = {2, 3};
  std::vector<std::array<long int, 2>> all_levels = {
      {2, 7}, {3, 6}, {4, 5}, {5, 4}, {6, 3}, {2, 6}, {3, 5}, {4, 4}, {5, 3}};
  std::vector<double> all_coefficients = {1, 1, 1, 1, 1, -1, -1, -1, -1};

  DVect resolution(std::array<long int, 2>{1L << 6, 1L << 7});
  auto global_dom =
      paliwa::optional_initialize_dims_periodic_unit_cube(resolution);

  std::array<int, 2> par_vector = {2, 2};
  auto [local_dom, cart_comm] = paliwa::decompose_domain_on_communicator(
      global_dom, sub_comm, par_vector);

  DVect ddc_min(minimum_level), ddc_max(maximum_level);
  std::string const wavelet = "biorthogonal";

  auto instances_arr = Kokkos::Experimental::partition_space(
      Kokkos::DefaultExecutionSpace(), 1, 1, 1, 1, 1, 1, 1, 1);
  std::vector<Kokkos::DefaultExecutionSpace> instances(instances_arr.begin(),
                                                       instances_arr.end());

  std::vector<SDDom> full_doms, local_doms;
  for (auto const &lv : all_levels) {
    auto f = paliwa::strided_domain_from_level<DimX, DimY>(lv, maximum_level);
    full_doms.push_back(f);
    local_doms.push_back(paliwa::restrict_strided_with_discrete(f, local_dom));
  }

  using ChunkType = ddc::Chunk<double, SDDom, ddc::HostAllocator<double>>;
  std::vector<ChunkType> grids;
  for (size_t i = 0; i < all_levels.size(); ++i) {
    grids.emplace_back("g" + std::to_string(i), local_doms[i],
                       ddc::HostAllocator<double>());
    auto s = grids.back().span_view();
    ddc::host_for_each(local_doms[i], [&](DElem e) {
      auto c = ddc::coordinate(e).array();
      s(e) = std::sin(pi * c[0]) * std::sin(pi * c[1]);
    });
    // Dump component grid before hierarchization
    std::string level_str;
    for (auto l : all_levels[i]) {
      level_str += std::to_string(l) + "_";
    }
    level_str += "2d";
    paliwa::dump_chunk_span_to_binary_file(
        s, "distributed_grid_" + level_str + "_rank" +
               std::to_string(world_rank) + ".raw");

    DVect lv(all_levels[i]);
    paliwa::distributed_hierarchize(s, full_doms[i], lv, ddc_min, ddc_max,
                                    wavelet, cart_comm,
                                    instances[i % instances.size()]);
  }
  paliwa::fence_all_instances(instances);

  auto full_max = paliwa::strided_domain_from_level<DimX, DimY>(maximum_level,
                                                                maximum_level);
  auto local_max = paliwa::restrict_strided_with_discrete(full_max, local_dom);

  ddc::Chunk combined("combined", local_max, ddc::HostAllocator<double>());
  auto cs = combined.span_view();
  ddc::host_for_each(local_max, [&](DElem e) { cs(e) = 0.0; });

  for (size_t i = 0; i < all_levels.size(); ++i) {
    double coeff = all_coefficients[i];
    auto gs = grids[i].span_cview();
    ddc::host_for_each(local_doms[i], [&](DElem e) { cs(e) += coeff * gs(e); });
  }

  paliwa::distributed_dehierarchize(cs, full_max, ddc_max, ddc_min, ddc_max,
                                    wavelet, cart_comm, instances[0]);
  paliwa::fence_all_instances(instances);

  double local_sum = 0.0;
  ddc::host_for_each(local_max, [&](DElem e) { local_sum += cs(e); });
  double global_sum = 0.0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, cart_comm);

  EXPECT_NEAR(global_sum / static_cast<double>(full_max.size()),
              sinusoid_integral_analytical(2), 0.031);

  MPI_Comm_free(&cart_comm);
  MPI_Comm_free(&sub_comm);
}

TEST(combination_technique, distributed_2d) {
  test_distributed_combination_technique_2d();
}

#endif // PALIWA_WITH_MPI