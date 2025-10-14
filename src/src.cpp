// SPDX-License-Identifier: MIT
// A lot reused from ddc/tests/strided_discrete_domain.cpp
// for these parts: Copyright (C) The DDC development team,
// see DDC's COPYRIGHT.md file

#include <algorithm>
#include <iostream>
#include <numeric>
#include <ranges>
#include <vector>

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

#include <Kokkos_Core.hpp>

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

#if defined(PERIODIC_DOMAIN)
static constexpr ddc::BoundCond BoundCond = ddc::BoundCond::PERIODIC;
template <class DDim>
using ExtrapolationRule = ddc::PeriodicExtrapolationRule<DDim>;
#else
static constexpr ddc::BoundCond BoundCond = ddc::BoundCond::GREVILLE;
template <class DDim> using ExtrapolationRule = ddc::NullExtrapolationRule;
#endif

template <class DDim>
using GrevillePoints =
    ddc::GrevilleInterpolationPoints<ddc::UniformBSplines<DDim, 1>, BoundCond,
                                     BoundCond>;
struct DDimX : GrevillePoints<X>::interpolation_discrete_dimension_type {};
// struct DDimX : ddc::UniformPointSampling<X>
// {
// };
using DElemX = ddc::DiscreteElement<DDimX>;
using DVectX = ddc::DiscreteVector<DDimX>;
using DDomX = ddc::StridedDiscreteDomain<DDimX>;

struct DDimY : GrevillePoints<Y>::interpolation_discrete_dimension_type {};
using DElemY = ddc::DiscreteElement<DDimY>;
using DVectY = ddc::DiscreteVector<DDimY>;
using DDomY = ddc::StridedDiscreteDomain<DDimY>;

#if DIMENSIONALITY > 2
struct DDimZ : GrevillePoints<Z>::interpolation_discrete_dimension_type {};
using DElemZ = ddc::DiscreteElement<DDimZ>;
using DVectZ = ddc::DiscreteVector<DDimZ>;
using DDomZ = ddc::StridedDiscreteDomain<DDimZ>;
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

using DElemZYX = ddc::DiscreteElement<DDimZ, DDimY, DDimX>;
using DVectZYX = ddc::DiscreteVector<DDimZ, DDimY, DDimX>;
using DDomZYX = ddc::DiscreteDomain<DDimZ, DDimY, DDimX>;
using SDDomZYX = ddc::StridedDiscreteDomain<DDimZ, DDimY, DDimX>;

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
                         [](long int ml) { return (1 << ml); });
  DVect resolution_all;
  ddc::detail::array(resolution_all) =
      resolution; // TODO temporary solution until assignment from std::array is
                  // implemented
  // TODO special case level 0
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
};
static const std::map<std::string,
                      std::vector<std::pair<int, std::array<double, 3>>>>
    lifting_wavelet_reconstruct_offsets_and_coefficients = {
        {"hat", {{1, {0.5, 1.0, 0.5}}}},
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
          typename DDomainType, // TODO either DDom or SDDom
          typename LevelRange>  // TODO input_range concept
void transform_in(DDomainType const &strided_domain,
                  ddc::ChunkSpan<double, DDomainType> const strided_grid,
                  std::array<long int, dimensionality> const &level,
                  std::array<long int, dimensionality> const &maximum_level,
                  LevelRange const &one_d_level_range,
                  std::vector<std::pair<int, std::array<double, 3>>> const
                      &lifting_offsets_and_coefficients) {
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
        throw std::runtime_error("Offset 0 not yet implemented");
      } else if (offset == 1) {
        coarsen_domain =
            odd_strided_domain_from_domain<DDimInWhichToHierarchize>;
      } else {
        throw std::runtime_error("Filter offset not supported");
      }
      auto const write_to_domain = coarsen_domain(operating_domain);
      ddc::parallel_for_each(
          write_to_domain, KOKKOS_LAMBDA(DElem const ixyz) {
            // how to access / slice at every other point in x?
            // check for out of bounds
            if (ddc::DiscreteElement<DDimInWhichToHierarchize>(ixyz) +
                    ddc::DiscreteVector<DDimInWhichToHierarchize>(
                        current_stride) <=
                ddc::DiscreteElement<DDimInWhichToHierarchize>(
                    write_to_domain.back())) {
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
                          ixyz - ddc::DiscreteVector<DDimInWhichToHierarchize>(
                                     current_stride)) +
                  filter[1] * strided_grid(ixyz) +
                  filter[2] * strided_grid(wraparound);
            }
          });
    }
  }
}

template <typename DDimInWhichToHierarchize, typename DDomainType>
void hierarchize_in(DDomainType const &strided_domain,
                    ddc::ChunkSpan<double, DDomainType> const strided_grid,
                    std::array<long int, dimensionality> const &level,
                    std::array<long int, dimensionality> const &minimum_level,
                    std::array<long int, dimensionality> const &maximum_level) {

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
      lifting_wavelet_filter_offsets_and_coefficients.at("hat"));
}

template <typename DDimInWhichToHierarchize, typename DDomainType>
void dehierarchize_in(
    DDomainType const &strided_domain,
    ddc::ChunkSpan<double, DDomainType> const strided_grid,
    std::array<long int, dimensionality> const &level,
    std::array<long int, dimensionality> const &minimum_level,
    std::array<long int, dimensionality> const &maximum_level) {

  auto const ddc_level_1d_vec =
      get_dimension_component<DDimInWhichToHierarchize>(level);
  auto const ddc_min_level_1d_vec =
      get_dimension_component<DDimInWhichToHierarchize>(minimum_level);

  auto increasing_range =
      std::views::iota(static_cast<long int>(ddc_min_level_1d_vec) - 1,
                       static_cast<long int>(ddc_level_1d_vec) + 1);
  return transform_in<DDimInWhichToHierarchize>(
      strided_domain, strided_grid, level, maximum_level, increasing_range,
      lifting_wavelet_reconstruct_offsets_and_coefficients.at("hat"));
}

template <typename T> // with T for example std::array<long int, dimensionality>
void iterate_hierarchical_subspaces(const T &nodal_level, T &tmp_level,
                                    size_t current_dim,
                                    std::function<void(const T &)> callback) {
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
  std::ofstream file(filename, std::ios::trunc | std::ios::binary);
  auto chunk_size = span.size();
  for (auto i = 0; i < chunk_size; ++i) {
    file.write(reinterpret_cast<char *>(&span.data_handle()[i]),
               sizeof(span.data_handle()[i]));
  }
}

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard const kokkos_scope;
  ddc::ScopeGuard const ddc_scope;

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
  std::vector<SDDom> component_grid_domains;
  // TODO these as Kokkos unordered_map?
  std::vector<ddc::Chunk<double, SDDom>> level_data;

  for (int grid_index = 0; grid_index < all_levels.size(); ++grid_index) {
    auto &level = all_levels[grid_index];
    component_grid_domains.emplace_back(
        strided_domain_from_level(level, maximum_level));
    level_data.emplace_back(ddc::Chunk(
        "strided_grid_" + std::to_string(grid_index),
        component_grid_domains.back(), ddc::DeviceAllocator<double>()));
    auto strided_grid = level_data.back().span_view();

    // initialize!
    ddc::parallel_for_each(
        component_grid_domains.back(), KOKKOS_LAMBDA(DElem const ixyz) {
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

  // hierarchize / wavelet-ify / filter in each direction
  for (int grid_index = 0; grid_index < all_levels.size(); ++grid_index) {
    // todo this for is another potential parallel_for_each!
    auto &level = all_levels[grid_index];
    SDDom const &strided_domain = component_grid_domains[grid_index];
    auto strided_grid = level_data[grid_index].span_view();

    hierarchize_in<DDimX>(strided_domain, strided_grid, level, minimum_level,
                          maximum_level);
    hierarchize_in<DDimY>(strided_domain, strided_grid, level, minimum_level,
                          maximum_level);
#if DIMENSIONALITY > 2
    hierarchize_in<DDimZ>(strided_domain, strided_grid, level, minimum_level,
                          maximum_level);
#endif
  }

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

  std::map<std::array<long int, dimensionality>, std::pair<SDDom, double *>>
      subspaces_domains_and_data_pointers; // TODO kokkos::unordered_map?
  size_t accumulated_size = 0;
  for (const auto &subspace_level_and_count : subspace_count) {
    auto const &subspace_level = subspace_level_and_count.first;
    auto const &count = subspace_level_and_count.second;
    if (count > 1) {
      auto subspace_domain =
          strided_hierarchical_domain_from_level(subspace_level, maximum_level);
      accumulated_size += subspace_domain.size();
      subspaces_domains_and_data_pointers[subspace_level] =
          std::make_pair(std::move(subspace_domain), nullptr);
    }
  }
  std::cout << "Total size of all subspaces: " << accumulated_size << std::endl;

  // allocate once
  std::vector<double> all_subspace_data(accumulated_size);
  size_t current_data_pointer_index = 0;
  for (auto &subspace_level_and_data : subspaces_domains_and_data_pointers) {
    auto &subspace_domain = subspace_level_and_data.second.first;
    auto &data_pointer = subspace_level_and_data.second.second;
    // basically exclusive scan
    data_pointer = all_subspace_data.data() + current_data_pointer_index;
    current_data_pointer_index += subspace_domain.size();
  }

  // collect component grids onto the sparse grid
  for (size_t grid_index = 0; grid_index < all_levels.size(); ++grid_index) {
    // todo this for is another potential parallel_for_each!
    auto &level = all_levels[grid_index];
    double coefficient = all_combi_coefficients[grid_index];
    SDDom const &strided_domain = component_grid_domains[grid_index];
    auto strided_grid = level_data[grid_index].span_view();

    for (const auto &subspace_level_and_data :
         subspaces_domains_and_data_pointers) {
      auto const &subspace_level = subspace_level_and_data.first;
      auto const &subspace_domain = subspace_level_and_data.second.first;
      auto const &data_pointer = subspace_level_and_data.second.second;
      bool contains = true; // TODO use ddc contains domain operator
      for (size_t d = 0; d < dimensionality; ++d) {
        if (subspace_level[d] > level[d]) {
          contains = false;
          break;
        }
      }
      if (contains) {
        // copy data into the allocated space
        ddc::ChunkSpan<double, SDDom> subspace_chunk_span(data_pointer,
                                                          subspace_domain);
        auto subspace_view = subspace_chunk_span.span_view();
        ddc::parallel_for_each(
            subspace_domain, KOKKOS_LAMBDA(DElem const ixyz) {
              subspace_view(ixyz) += coefficient * strided_grid(ixyz);
            });
      }
    }
  }

  // interpolate all onto full grid
  // allocate and initialize to 0
  ddc::Chunk full_grid("interpolated_on_full_grid", dom_all,
                       ddc::DeviceAllocator<double>());
  auto full_grid_view = full_grid.span_view();

  // now copy into full grid
  for (const auto &subspace_level_and_data :
       subspaces_domains_and_data_pointers) {
    auto const &subspace_level = subspace_level_and_data.first;
    auto const &subspace_domain = subspace_level_and_data.second.first;
    auto const &data_pointer = subspace_level_and_data.second.second;
    ddc::ChunkSpan<double, SDDom> subspace_chunk_span(data_pointer,
                                                      subspace_domain);
    auto subspace_view = subspace_chunk_span.span_view();
    ddc::parallel_for_each(
        subspace_domain, KOKKOS_LAMBDA(DElem const ixyz) {
          full_grid_view(ixyz) += subspace_view(ixyz);
        });
  }

  //   de-hierarchize on the combined full grid
  dehierarchize_in<DDimX>(dom_all, full_grid_view, maximum_level, minimum_level,
                          maximum_level);
  dehierarchize_in<DDimY>(dom_all, full_grid_view, maximum_level, minimum_level,
                          maximum_level);

  std::string max_level_str = "";
  for (auto l : maximum_level) {
    max_level_str += std::to_string(l) + "_";
  }
  max_level_str += std::to_string(dimensionality) + "d";
  std::string const filename = "full_grid_" + max_level_str + ".raw";
  dump_chunk_span_to_binary_file(full_grid_view, filename);
}