// SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <cstddef>
#include <ddc/ddc.hpp>
#include <ranges>
#include <utility>
#include <vector>

#include "paliwa_domains.hpp"
#include "paliwa_wavelets.hpp"

namespace paliwa {

// CPU pole packing: gather per pass, or once for the complete level sequence.
enum class PoleExecution { Direct, PackedPasses, PackedPoles };

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

template <typename Dim, typename Chunk, typename RemoteChunk,
          typename ExecSpace, typename... Dims>
void transform_packed_poles(
    Chunk grid, RemoteChunk remote_grid,
    ddc::StridedDiscreteDomain<Dims...> const &outputs,
    std::vector<long> const &levels, long level, long maximum_level,
    std::vector<std::pair<int, std::array<double, 3>>> const &lifting_passes,
    std::vector<std::vector<PeriodicBounds>> const &writes,
    std::vector<long> const &inputs, PoleExecution order, ExecSpace instance) {
  using Value = std::remove_cv_t<typename Chunk::element_type>;
  using Element = ddc::DiscreteElement<Dims...>;
  long const period = 1L << maximum_level;
  long const spacing_log2 = maximum_level - level;
  auto const cover = covering_periodic_bounds(inputs, period);
  long const width =
      cover.max - cover.min + (cover.min > cover.max ? period : 0);
  auto slot = [&](long x) {
    long const relative = x - cover.min;
    return (relative < 0 ? relative + period : relative) >> spacing_log2;
  };
  Kokkos::Experimental::UniqueToken<ExecSpace> tokens(instance);
  Kokkos::View<Value **, Kokkos::LayoutRight, typename ExecSpace::memory_space>
      buffers(
          Kokkos::view_alloc(instance, Kokkos::WithoutInitializing, "poles"),
          tokens.size(), (width >> spacing_log2) + 1);
  auto const poles = ddc::remove_dims_of<Dim>(outputs);
  auto const output_axis = ddc::select<Dim>(outputs);

  // Enumerate exactly the planned writes, including intervals crossing zero.
  auto visit_writes = [&](size_t l, size_t p, auto visit) {
    auto const b = writes[l][p];
    if (b.empty())
      return;
    long const stride = 1L << (maximum_level - levels[l]);
    long const last = b.max + (b.min > b.max ? period : 0);
    for (long unwrapped = b.min; unwrapped <= last; unwrapped += 2 * stride) {
      long const x = unwrapped >= period ? unwrapped - period : unwrapped;
      visit(x, x == 0 ? period - stride : x - stride,
            x + stride == period ? 0 : x + stride);
    }
  };
  // Translate each pass once. The period is the full grid period in pole
  // slots, not the length of the potentially shorter interval allocation.
  std::vector<std::vector<PolePassSegment>> packed_indices;
  packed_indices.reserve(levels.size() * lifting_passes.size());
  for (size_t l = 0; l < levels.size(); ++l)
    for (size_t p = 0; p < lifting_passes.size(); ++p)
      packed_indices.push_back(pole_pass_segments(
          writes[l][p], cover.min, level, maximum_level, levels[l]));
  auto apply_pass = [&](int buffer, size_t l, size_t p) {
    auto const &coefficients = lifting_passes[p].second;
    for (auto const &segment : packed_indices[l * lifting_passes.size() + p]) {
      auto x = segment.first;
      auto lower = segment.lower;
      auto upper = segment.upper;
      for (auto remaining = segment.count; remaining > 0; --remaining) {
        buffers(buffer, x) =
            static_cast<Value>(coefficients[0]) * buffers(buffer, lower) +
            static_cast<Value>(coefficients[1]) * buffers(buffer, x) +
            static_cast<Value>(coefficients[2]) * buffers(buffer, upper);
        x += segment.stride;
        lower += segment.stride;
        upper += segment.stride;
      }
    }
  };
  if (order == PoleExecution::PackedPoles) {
    // Resolve coordinate-to-storage mappings once per axis, not per value of
    // every pole. Local and remote values remain in separate allocations.
    auto make_copy = [&](auto source, std::vector<long> const &indices) {
      struct Offset {
        std::ptrdiff_t source;
        long scratch;
      };
      std::vector<Offset> offsets;
      offsets.reserve(indices.size());
      long const first = indices.empty() ? 0 : indices.front();
      std::ptrdiff_t anchor_offset = 0;
      if (!indices.empty()) {
        auto const anchor = poles.front();
        anchor_offset =
            &source(Element(anchor, ddc::DiscreteElement<Dim>(first))) -
            source.data_handle();
        for (long x : indices)
          offsets.push_back(
              {&source(Element(anchor, ddc::DiscreteElement<Dim>(x))) -
                   source.data_handle(),
               slot(x)});
      }
      return [=, offsets = std::move(offsets)](auto pole, int buffer,
                                               bool scatter) {
        if (offsets.empty())
          return;
        auto const data = source.data_handle();
        auto const base =
            &source(Element(pole, ddc::DiscreteElement<Dim>(first))) - data -
            anchor_offset;
        if (scatter) {
          for (auto const &offset : offsets)
            data[base + offset.source] = buffers(buffer, offset.scratch);
        } else {
          for (auto const &offset : offsets)
            buffers(buffer, offset.scratch) = data[base + offset.source];
        }
      };
    };
    std::vector<long> local_inputs, remote_inputs, output_indices;
    auto const local_axis = ddc::select<Dim>(grid.domain());
    for (long x : inputs) {
      if (local_axis.contains(ddc::DiscreteElement<Dim>(x)))
        local_inputs.push_back(x);
      else
        remote_inputs.push_back(x);
    }
    ddc::host_for_each(output_axis, [&](auto x) {
      output_indices.push_back(x.template uid<Dim>());
    });
    auto const gather_local = make_copy(grid, local_inputs);
    auto const gather_remote = make_copy(remote_grid, remote_inputs);
    auto const scatter_local = make_copy(grid, output_indices);
    ddc::parallel_for_each(instance, poles, [&](auto pole) {
      int const buffer = tokens.acquire();
      gather_local(pole, buffer, false);
      gather_remote(pole, buffer, false);
      for (size_t l = 0; l < levels.size(); ++l)
        for (size_t p = 0; p < lifting_passes.size(); ++p)
          apply_pass(buffer, l, p);
      scatter_local(pole, buffer, true);
      tokens.release(buffer);
    });
    // Copy plans must outlive all uses by the execution space.
    instance.fence();
  } else {
    for (size_t l = 0; l < levels.size(); ++l) {
      for (size_t p = 0; p < lifting_passes.size(); ++p) {
        if (writes[l][p].empty())
          continue;
        ddc::parallel_for_each(instance, poles, [&](auto pole) {
          int const buffer = tokens.acquire();
          visit_writes(l, p, [&](long x, long lower, long upper) {
            for (long input : {lower, x, upper})
              buffers(buffer, slot(input)) =
                  grid(Element(pole, ddc::DiscreteElement<Dim>(input)));
          });
          apply_pass(buffer, l, p);
          visit_writes(l, p, [&](long x, long, long) {
            grid(Element(pole, ddc::DiscreteElement<Dim>(x))) =
                buffers(buffer, slot(x));
          });
          tokens.release(buffer);
        });
        instance.fence();
      }
    }
  }
  // Complete before releasing buffers and host plan references.
  instance.fence();
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

namespace detail {

// Plan exact writes for the requested output domain.
// Packed poles may read required inputs from separate local/remote storage.
// Other execution modes require all inputs in grid.
template <typename Dim, typename Chunk, typename RemoteChunk,
          typename LevelRange, typename ExecSpace, typename... Dims>
constexpr bool transform_with_remote(
    Chunk grid, RemoteChunk remote_grid,
    ddc::DiscreteVector<Dims...> const &level,
    ddc::DiscreteVector<Dims...> const &maximum_level, LevelRange const &levels,
    std::vector<std::pair<int, std::array<double, 3>>> const &filters,
    ddc::StridedDiscreteDomain<Dims...> const &outputs, ExecSpace instance,
    PoleExecution pole_execution) {
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
  std::vector<long> inputs;
  for_each_transform_index<Dim>(bounds, max_level, [&](auto i) {
    if (!ddc::select<Dim>(grid.domain()).contains(i) &&
        !(pole_execution == PoleExecution::PackedPoles &&
          ddc::select<Dim>(remote_grid.domain()).contains(i)))
      throw std::runtime_error(
          "Transform domain is missing required input values");
    if (pole_execution != PoleExecution::Direct)
      inputs.push_back(i.template uid<Dim>());
  });
  if constexpr (Kokkos::SpaceAccessibility<ExecSpace,
                                           Kokkos::HostSpace>::accessible) {
    if (pole_execution != PoleExecution::Direct && !ordered_levels.empty() &&
        !filters.empty()) {
      std::sort(inputs.begin(), inputs.end());
      detail::transform_packed_poles<Dim>(
          grid, remote_grid, outputs, ordered_levels,
          static_cast<long>(ddc::select<Dim>(level)), max_level, filters,
          writes_by_level, inputs, pole_execution, instance);
      return true;
    }
  }
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

} // namespace detail

// Transform a requested output domain held in a single allocation.
template <typename Dim, typename Chunk, typename LevelRange, typename ExecSpace,
          typename... Dims>
constexpr bool transform_in(
    Chunk grid, ddc::DiscreteVector<Dims...> const &level,
    ddc::DiscreteVector<Dims...> const &maximum_level, LevelRange const &levels,
    std::vector<std::pair<int, std::array<double, 3>>> const &filters,
    ddc::StridedDiscreteDomain<Dims...> const &outputs, ExecSpace instance,
    PoleExecution pole_execution = PoleExecution::Direct) {
  return detail::transform_with_remote<Dim>(grid, grid, level, maximum_level,
                                            levels, filters, outputs, instance,
                                            pole_execution);
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
