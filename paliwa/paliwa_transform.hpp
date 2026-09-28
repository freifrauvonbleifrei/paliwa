// SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <ddc/ddc.hpp>
#include <ranges>
#include <vector>

#include "paliwa_domains.hpp"
#include "paliwa_wavelets.hpp"

namespace paliwa {

namespace detail {
template <bool CheckBounds, typename Dim, typename Chunk, typename Domain,
          typename ExecSpace>
void lifting_pass(Chunk grid, Domain const &writes, long stride, long period,
                  std::array<double, 3> const &filter, ExecSpace instance) {
  auto const storage = grid.domain();
  ddc::parallel_for_each(
      instance, writes, KOKKOS_LAMBDA(auto i) {
        if constexpr (CheckBounds) {
          if (!storage.contains(i))
            return;
        }
        auto lower = i;
        auto upper = i;
        long const x = i.template uid<Dim>();
        lower = replace_dim(lower, ddc::DiscreteElement<Dim>(
                                       x == 0 ? period - stride : x - stride));
        upper = replace_dim(upper, ddc::DiscreteElement<Dim>(
                                       x + stride == period ? 0 : x + stride));
        if constexpr (CheckBounds) {
          if (!storage.contains(lower) || !storage.contains(upper))
            return;
        }
        using Value = std::decay_t<decltype(grid(i))>;
        grid(i) = static_cast<Value>(filter[0]) * grid(lower) +
                  static_cast<Value>(filter[1]) * grid(i) +
                  static_cast<Value>(filter[2]) * grid(upper);
      });
}
} // namespace detail

// Existing API: complete strided grids need no membership checks. Partial
// buffers without an explicit output domain retain their checked behavior.
template <typename Dim, typename Chunk, typename LevelRange, typename ExecSpace,
          typename... Dims>
constexpr bool
transform_in(Chunk grid, ddc::DiscreteVector<Dims...> const &level,
             ddc::DiscreteVector<Dims...> const &maximum_level,
             LevelRange const &levels,
             std::vector<std::pair<int, std::array<double, 3>>> const &filters,
             ExecSpace instance = ExecSpace()) {
  using Domain = ddc::StridedDiscreteDomain<Dims...>;
  auto const full = strided_domain_from_level<Dims...>(
      ddc::detail::array(level), ddc::detail::array(maximum_level));
  bool complete = false;
  if constexpr (std::is_same_v<std::decay_t<decltype(grid.domain())>, Domain>)
    complete = grid.domain() == full;
  auto [even, odd] = get_even_and_odd_half_domain_functors<Dim, Dims...>();
  auto current = level;
  for (long l : levels) {
    assert(l > 0);
    current.template get<Dim>() = l;
    auto const operating = strided_domain_from_level<Dims...>(
        ddc::detail::array(current), ddc::detail::array(maximum_level));
    long const stride = operating.strides().template get<Dim>();
    long const period = stride * operating.extents().template get<Dim>();
    for (auto const &[offset, filter] : filters) {
      if (offset != 0 && offset != 1)
        throw std::runtime_error("Filter offset not supported");
      auto const writes = offset == 0 ? even(operating) : odd(operating);
      if (complete)
        detail::lifting_pass<false, Dim>(grid, writes, stride, period, filter,
                                         instance);
      else
        detail::lifting_pass<true, Dim>(grid, writes, stride, period, filter,
                                        instance);
    }
  }
  return true;
}

// Plan exact writes for the requested output domain.
// Storage must contain the exact required inputs computed below.
template <typename Dim, typename Chunk, typename LevelRange, typename ExecSpace,
          typename... Dims>
constexpr bool transform_in(
    Chunk grid, ddc::DiscreteVector<Dims...> const &level,
    ddc::DiscreteVector<Dims...> const &maximum_level, LevelRange const &levels,
    std::vector<std::pair<int, std::array<double, 3>>> const &filters,
    ddc::StridedDiscreteDomain<Dims...> const &outputs, ExecSpace instance) {
  auto const local_axis = ddc::select<Dim>(outputs);
  if (outputs.empty())
    return true;
  long const max_level = static_cast<long>(ddc::select<Dim>(maximum_level));
  long const period = 1L << max_level;
  std::vector<long> ordered_levels(levels.begin(), levels.end());
  std::vector<std::vector<PeriodicBounds>> writes_by_level(
      ordered_levels.size(), std::vector<PeriodicBounds>(filters.size()));
  auto const bounds = transform_bounds(
      {static_cast<long>(local_axis.front().template uid<Dim>()),
       static_cast<long>(local_axis.back().template uid<Dim>())},
      static_cast<long>(ddc::select<Dim>(level)), max_level, ordered_levels,
      filters, [&](size_t i, size_t pass, PeriodicBounds writes) {
        writes_by_level[i][pass] = writes;
      });
  for_each_transform_index<Dim>(bounds, max_level, [&](auto i) {
    if (!ddc::select<Dim>(grid.domain()).contains(i))
      throw std::runtime_error(
          "Transform domain is missing required input values");
  });
  auto [even, odd] = get_even_and_odd_half_domain_functors<Dim, Dims...>();
  auto current = level;
  size_t index = 0;
  for (long l : levels) {
    current.template get<Dim>() = l;
    auto const operating = strided_domain_from_level<Dims...>(
        ddc::detail::array(current), ddc::detail::array(maximum_level));
    long const stride = 1L << (max_level - l);
    for (size_t pass = 0; pass < filters.size(); ++pass) {
      auto const &[offset, filter] = filters[pass];
      if (offset != 0 && offset != 1)
        throw std::runtime_error("Filter offset not supported");
      auto const writes = local_pole_domain<Dim>(
          offset == 0 ? even(operating) : odd(operating), outputs);
      auto const pass_bounds = writes_by_level[index][pass];
      for (auto const &part :
           restrict_periodic_bounds<Dim>(writes, pass_bounds, period)) {
        detail::lifting_pass<false, Dim>(grid, part, stride, period, filter,
                                         instance);
      }
    }
    ++index;
  }
  return true;
}

template <typename DDimInWhichToHierarchize, typename ChunkSpanType,
          typename ExecSpace, // = Kokkos::DefaultExecutionSpace
          typename... DDims>
constexpr bool
hierarchize_in(ChunkSpanType const strided_grid,
               ddc::DiscreteVector<DDims...> const &level,
               ddc::DiscreteVector<DDims...> const &minimum_level,
               ddc::DiscreteVector<DDims...> const &maximum_level,
               std::string const &wavelet_name = "hat",
               ExecSpace instance = ExecSpace()) {
  auto const ddc_level_1d_vec = ddc::select<DDimInWhichToHierarchize>(level);
  auto const ddc_min_level_1d_vec =
      ddc::select<DDimInWhichToHierarchize>(minimum_level);
  assert(ddc_level_1d_vec >= ddc_min_level_1d_vec);
  assert(ddc_min_level_1d_vec >= 0);
  assert(ddc_level_1d_vec <=
         ddc::select<DDimInWhichToHierarchize>(maximum_level));

  auto decreasing_range =
      std::views::iota(static_cast<long int>(ddc_min_level_1d_vec + 1),
                       static_cast<long int>(ddc_level_1d_vec) + 1) |
      std::views::reverse;
  return transform_in<DDimInWhichToHierarchize>(
      strided_grid, level, maximum_level, decreasing_range,
      lifting_wavelet_filter_offsets_and_coefficients.at(wavelet_name),
      instance);
}

template <typename DDimInWhichToHierarchize, typename ChunkSpanType,
          typename ExecSpace, // = Kokkos::DefaultExecutionSpace
          typename... DDims>
constexpr bool
dehierarchize_in(ChunkSpanType const strided_grid,
                 ddc::DiscreteVector<DDims...> const &level,
                 ddc::DiscreteVector<DDims...> const &minimum_level,
                 ddc::DiscreteVector<DDims...> const &maximum_level,
                 std::string const &wavelet_name = "hat",
                 ExecSpace instance = ExecSpace()) {
  auto const ddc_level_1d_vec = ddc::select<DDimInWhichToHierarchize>(level);
  auto const ddc_min_level_1d_vec =
      ddc::select<DDimInWhichToHierarchize>(minimum_level);
  assert(ddc_level_1d_vec >= ddc_min_level_1d_vec);
  assert(ddc_min_level_1d_vec >= 0);
  assert(ddc_level_1d_vec <=
         ddc::select<DDimInWhichToHierarchize>(maximum_level));

  auto increasing_range =
      std::views::iota(static_cast<long int>(ddc_min_level_1d_vec + 1),
                       static_cast<long int>(ddc_level_1d_vec) + 1);
  return transform_in<DDimInWhichToHierarchize>(
      strided_grid, level, maximum_level, increasing_range,
      lifting_wavelet_reconstruct_offsets_and_coefficients.at(wavelet_name),
      instance);
}

template <typename ChunkSpanType,
          typename ExecSpace, // = Kokkos::DefaultExecutionSpace
          typename... DDims>
constexpr void hierarchize(ChunkSpanType const strided_grid,
                           ddc::DiscreteVector<DDims...> const &level,
                           ddc::DiscreteVector<DDims...> const &minimum_level,
                           ddc::DiscreteVector<DDims...> const &maximum_level,
                           std::string const &wavelet_name = "hat",
                           ExecSpace instance = ExecSpace()) {
  // fold expression to call for every dimension
  [[maybe_unused]] bool unused =
      (hierarchize_in<DDims>(strided_grid, level, minimum_level, maximum_level,
                             wavelet_name, instance) &&
       ...);
}

template <typename ChunkSpanType,
          typename ExecSpace, // = Kokkos::DefaultExecutionSpace
          typename... DDims>
constexpr void dehierarchize(ChunkSpanType const strided_grid,
                             ddc::DiscreteVector<DDims...> const &level,
                             ddc::DiscreteVector<DDims...> const &minimum_level,
                             ddc::DiscreteVector<DDims...> const &maximum_level,
                             std::string const &wavelet_name = "hat",
                             ExecSpace instance = ExecSpace()) {
  // fold expression to call for every dimension
  [[maybe_unused]] bool unused =
      (dehierarchize_in<DDims>(strided_grid, level, minimum_level,
                               maximum_level, wavelet_name, instance) &&
       ...);
}

} // namespace paliwa
