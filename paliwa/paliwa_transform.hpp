// SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include <array>
#include <functional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <ddc/ddc.hpp>

#include "paliwa_adapter.hpp"
#include "paliwa_distribute.hpp"
#include "paliwa_domains.hpp"
#include "paliwa_wavelets.hpp"

namespace paliwa {

template <typename DDimInWhichToTransform, typename ChunkSpanType,
          typename LevelRange, typename ExecSpace, typename... DDims>
constexpr bool
transform_in(ChunkSpanType const strided_grid,
             ddc::DiscreteVector<DDims...> const &level,
             ddc::DiscreteVector<DDims...> const &maximum_level,
             LevelRange const &one_d_level_range,
             std::vector<std::pair<int, std::array<double, 3>>> const
                 &lifting_offsets_and_coefficients,
             ExecSpace instance = ExecSpace()) {

  auto chunk_domain = strided_grid.domain();
  using DElem = ddc::DiscreteElement<DDims...>;
  using SDDom = ddc::StridedDiscreteDomain<DDims...>;

  auto [even_domain_functor, odd_domain_functor] =
      get_even_and_odd_half_domain_functors<DDimInWhichToTransform, DDims...>();

  ddc::DiscreteVector<DDims...> current_level(level);

  for (long int current_1d_level : one_d_level_range) {
    assert(current_1d_level > 0);
    current_level.template get<DDimInWhichToTransform>() = current_1d_level;

    auto const operating_domain = strided_domain_from_level<DDims...>(
        ddc::detail::array(current_level), ddc::detail::array(maximum_level));
    auto const current_stride =
        operating_domain.strides().template get<DDimInWhichToTransform>();
    auto const virtual_length = ddc::DiscreteVector<DDimInWhichToTransform>(
        operating_domain.extents().template get<DDimInWhichToTransform>() *
        current_stride);
    auto const this_d_stride =
        ddc::DiscreteVector<DDimInWhichToTransform>(current_stride);

    for (auto const &[offset, filter] : lifting_offsets_and_coefficients) {
      std::function<SDDom(SDDom const &)> coarsen_domain;
      if (offset == 0)
        coarsen_domain = even_domain_functor;
      else if (offset == 1)
        coarsen_domain = odd_domain_functor;
      else
        throw std::runtime_error("Filter offset not supported");

      auto const write_to_domain = coarsen_domain(operating_domain);

      ddc::parallel_for_each(
          instance, write_to_domain, KOKKOS_LAMBDA(DElem const ixyz) {
            // TODO: remove these checks for efficiency. Can't be deleted.
            if (!chunk_domain.contains(ixyz)) {
              return;
            }

            DElem lower_element = ixyz - this_d_stride;
            DElem upper_element = ixyz + this_d_stride;

            if ((offset == 1) &&
                (ddc::DiscreteElement<DDimInWhichToTransform>(ixyz) +
                     this_d_stride >
                 ddc::DiscreteElement<DDimInWhichToTransform>(
                     write_to_domain.back()))) {
              upper_element -= virtual_length;
            } else if ((offset == 0) &&
                       (ddc::DiscreteElement<DDimInWhichToTransform>(ixyz) <=
                        ddc::select<DDimInWhichToTransform>(
                            operating_domain.front()))) {
              lower_element += virtual_length;
            }

            if (!chunk_domain.contains(lower_element) ||
                !chunk_domain.contains(upper_element))
              return;

            using value_type = std::decay_t<decltype(strided_grid(ixyz))>;
            strided_grid(ixyz) =
                static_cast<value_type>(filter[0]) *
                    strided_grid(lower_element) +
                static_cast<value_type>(filter[1]) * strided_grid(ixyz) +
                static_cast<value_type>(filter[2]) *
                    strided_grid(upper_element);
          });
    }
  }
  return true;
}

template <typename DDimInWhichToHierarchize, typename ChunkSpanType,
          typename ExecSpace, typename... DDims>
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
          typename ExecSpace, typename... DDims>
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

template <typename DDim, typename ChunkSpanType, typename LevelRange,
          typename ExecSpace = Kokkos::DefaultHostExecutionSpace>
constexpr void
transform_mask(ChunkSpanType const strided_grid,
               ddc::DiscreteVector<DDim> const &level,
               ddc::DiscreteVector<DDim> const &maximum_level,
               LevelRange const &one_d_level_range,
               std::vector<std::pair<int, std::array<double, 3>>> const
                   &lifting_offsets_and_coefficients,
               ExecSpace instance = ExecSpace()) {
  using DElem = ddc::DiscreteElement<DDim>;
  using SDDom = ddc::StridedDiscreteDomain<DDim>;

  auto [even_domain_functor, odd_domain_functor] =
      get_even_and_odd_half_domain_functors<DDim, DDim>();

  ddc::DiscreteVector<DDim> current_level(level);

  for (long int current_1d_level : one_d_level_range) {
    assert(current_1d_level > 0);
    current_level.template get<DDim>() = current_1d_level;

    auto const operating_domain = strided_domain_from_level<DDim>(
        ddc::detail::array(current_level), ddc::detail::array(maximum_level));
    auto const current_stride = operating_domain.strides();
    auto const virtual_length =
        ddc::DiscreteVector<DDim>(operating_domain.extents() * current_stride);
    auto const this_d_stride = ddc::DiscreteVector<DDim>(current_stride);

    for (auto const &[offset, filter] :
         lifting_offsets_and_coefficients | std::ranges::views::reverse) {
      std::function<SDDom(SDDom const &)> coarsen_domain;
      if (offset == 0)
        coarsen_domain = even_domain_functor;
      else if (offset == 1)
        coarsen_domain = odd_domain_functor;
      else
        throw std::runtime_error("Filter offset not supported");

      auto const read_from_domain = coarsen_domain(operating_domain);

      ddc::parallel_for_each(
          instance, read_from_domain, KOKKOS_LAMBDA(DElem const ixyz) {
            if (strided_grid(ixyz) == 0.0)
              return;

            DElem lower_element = ixyz - this_d_stride;
            DElem upper_element = ixyz + this_d_stride;

            if ((offset == 1) &&
                (ixyz + this_d_stride > read_from_domain.back()))
              upper_element -= virtual_length;
            else if ((offset == 0) && (ixyz <= operating_domain.front()))
              lower_element += virtual_length;

            strided_grid(lower_element) +=
                std::abs(filter[0]) * strided_grid(ixyz);
            strided_grid(ixyz) += std::abs(filter[1]) * strided_grid(ixyz);
            strided_grid(upper_element) +=
                std::abs(filter[2]) * strided_grid(ixyz);
          });
    }
  }
}

template <typename SelectedDim, typename SDDom, typename DDom>
constexpr ddc::SparseDiscreteDomain<SelectedDim> get_required_transform_domain(
    bool is_for_hierarchization, SDDom const &full_domain,
    DDom const &local_domain, ddc::DiscreteVector<SelectedDim> const &level,
    ddc::DiscreteVector<SelectedDim> const &minimum_level,
    ddc::DiscreteVector<SelectedDim> const &maximum_level,
    std::string const &wavelet_name = "hat") {
  using DElem = ddc::DiscreteElement<SelectedDim>;
  using ExecSpace = Kokkos::DefaultExecutionSpace;

  ddc::Chunk full_pole_chunk(
      "full_pole_chunk", full_domain,
      ddc::KokkosAllocator<float, ExecSpace::memory_space>());
  auto full_pole = full_pole_chunk.span_view();

  ddc::parallel_for_each(
      ExecSpace{}, full_domain,
      KOKKOS_LAMBDA(DElem ixyz) { full_pole(ixyz) = 0.0f; });
  ddc::parallel_for_each(
      ExecSpace{}, local_domain,
      KOKKOS_LAMBDA(DElem ixyz) { full_pole(ixyz) = 1.0f; });

  auto increasing_range =
      std::views::iota(static_cast<long int>(minimum_level + 1),
                       static_cast<long int>(level) + 1);
  if (is_for_hierarchization) {
    transform_mask<SelectedDim>(
        full_pole, level, maximum_level, increasing_range,
        lifting_wavelet_filter_offsets_and_coefficients.at(wavelet_name),
        ExecSpace{});
  } else {
    transform_mask<SelectedDim>(
        full_pole, level, maximum_level, increasing_range | std::views::reverse,
        lifting_wavelet_reconstruct_offsets_and_coefficients.at(wavelet_name),
        ExecSpace{});
  }
  ddc::parallel_for_each(
      ExecSpace{}, local_domain,
      KOKKOS_LAMBDA(DElem ixyz) { full_pole(ixyz) = 0.0f; });

  // ---- Stream compaction, stride-aware ----
  int const n = static_cast<int>(full_domain.size());
  auto const stride = full_domain.strides().template get<SelectedDim>();
  DElem const front = full_domain.front();

  int const count = ddc::parallel_transform_reduce(
      ExecSpace{}, full_domain, 0, ddc::reducer::sum<int>(),
      KOKKOS_LAMBDA(DElem ixyz)->int {
        return full_pole(ixyz) != 0.0f ? 1 : 0;
      });

  // No ddc wrapper for scan; stays as raw Kokkos, but kept
  // stride-aware/DElem-based.
  Kokkos::View<DElem *, ExecSpace::memory_space> required_device(
      "required_elements_device", count);
  Kokkos::parallel_scan(
      "compact_ghost", Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(int i, int &update, bool final) {
        DElem elem = front + ddc::DiscreteVector<SelectedDim>(
                                 static_cast<long int>(i) * stride);
        if (full_pole(elem) != 0.0f) {
          if (final)
            required_device(update) = elem;
          ++update;
        }
      });

  Kokkos::View<DElem *, Kokkos::SharedSpace> required_shared(
      "required_elements_shared", count);
  Kokkos::deep_copy(required_shared, required_device);
  return ddc::SparseDiscreteDomain<SelectedDim>(required_shared);
}

namespace detail {

template <typename DimToTransform, bool IsHierarchization,
          typename ChunkSpanType, typename ExecSpace, typename... DDims>
bool distributed_transform_in(
    ChunkSpanType const local_grid,
    ddc::StridedDiscreteDomain<DDims...> const &full_strided_domain,
    ddc::DiscreteVector<DDims...> const &level,
    ddc::DiscreteVector<DDims...> const &minimum_level,
    ddc::DiscreteVector<DDims...> const &maximum_level,
    std::string const &wavelet_name, int dim_index, ExecSpace instance) {
  using value_type = typename ChunkSpanType::element_type;
  using DElem1d = ddc::DiscreteElement<DimToTransform>;

  auto local_domain = local_grid.domain();
  auto full_1d_strided = ddc::select<DimToTransform>(full_strided_domain);

  auto local_restricted_1d = restrict_strided_with_discrete(
      full_1d_strided, ddc::select<DimToTransform>(local_domain));

  auto ghost_domain_1d = get_required_transform_domain<DimToTransform>(
      IsHierarchization, full_1d_strided, local_restricted_1d,
      ddc::select<DimToTransform>(level),
      ddc::select<DimToTransform>(minimum_level),
      ddc::select<DimToTransform>(maximum_level), wavelet_name);

  if (ghost_domain_1d.size() == 0) {
    if constexpr (IsHierarchization)
      return hierarchize_in<DimToTransform>(local_grid, level, minimum_level,
                                            maximum_level, wavelet_name,
                                            instance);
    else
      return dehierarchize_in<DimToTransform>(local_grid, level, minimum_level,
                                              maximum_level, wavelet_name,
                                              instance);
  }

  auto global_size_1d =
      full_1d_strided.extents().template get<DimToTransform>() *
      full_1d_strided.strides().template get<DimToTransform>();
  ddc::DiscreteDomain<DimToTransform> global_domain_1d(
      DElem1d(full_1d_strided.front()),
      ddc::DiscreteVector<DimToTransform>(global_size_1d));

  auto extended_1d = union_of_sparse_domains(
      sparse_from_strided_domain(local_restricted_1d), ghost_domain_1d);

  ddc::SparseDiscreteDomain<DDims...> extended_domain(
      [&]() -> ddc::SparseDiscreteDomain<DDims> {
        if constexpr (std::is_same_v<DDims, DimToTransform>)
          return extended_1d;
        else
          return sparse_from_strided_domain(restrict_strided_with_discrete(
              ddc::select<DDims>(full_strided_domain),
              ddc::select<DDims>(local_domain)));
      }()...);

  ddc::Chunk extended_chunk(
      "extended_buffer", extended_domain,
      ddc::KokkosAllocator<value_type, typename ExecSpace::memory_space>());
  auto extended_span = extended_chunk.span_view();

  auto local_restricted =
      restrict_strided_with_discrete(full_strided_domain, local_domain);
  ddc::parallel_for_each(
      instance, local_restricted,
      KOKKOS_LAMBDA(ddc::DiscreteElement<DDims...> elem) {
        if (!local_grid.domain().contains(elem))
          return;
        if (!extended_span.domain().contains(elem))
          return;
        extended_span(elem) = local_grid(elem);
      });

  auto remote_ghost_fn = [&](ddc::StridedDiscreteDomain<DimToTransform> const
                                 &remote_restricted_1d) {
    return get_required_transform_domain<DimToTransform>(
        IsHierarchization, full_1d_strided, remote_restricted_1d,
        ddc::select<DimToTransform>(level),
        ddc::select<DimToTransform>(minimum_level),
        ddc::select<DimToTransform>(maximum_level), wavelet_name);
  };

  auto peers = compute_peer_exchanges<DimToTransform, IsHierarchization>(
      classify_ghost_by_rank<DimToTransform>(ghost_domain_1d, global_domain_1d,
                                             dim_index),
      full_1d_strided, global_domain_1d,
      ddc::select<DimToTransform>(local_domain), dim_index, remote_ghost_fn);

  exchange_ghost_slices<DimToTransform>(peers, local_grid, extended_span,
                                        local_restricted, local_restricted_1d);

  if constexpr (IsHierarchization)
    hierarchize_in<DimToTransform>(extended_span, level, minimum_level,
                                   maximum_level, wavelet_name, instance);
  else
    dehierarchize_in<DimToTransform>(extended_span, level, minimum_level,
                                     maximum_level, wavelet_name, instance);

  ddc::parallel_for_each(
      instance, local_restricted,
      KOKKOS_LAMBDA(ddc::DiscreteElement<DDims...> elem) {
        if (!local_grid.domain().contains(elem))
          return;
        if (!extended_span.domain().contains(elem))
          return;
        local_grid(elem) = extended_span(elem);
      });

  return true;
}

// Index-sequence helpers to fold over all dimensions.

template <typename ChunkSpanType, typename ExecSpace, typename... DDims,
          std::size_t... Is>
void hierarchize_all_dims(
    ChunkSpanType const local_grid,
    ddc::StridedDiscreteDomain<DDims...> const &full_strided_domain,
    ddc::DiscreteVector<DDims...> const &level,
    ddc::DiscreteVector<DDims...> const &minimum_level,
    ddc::DiscreteVector<DDims...> const &maximum_level,
    std::string const &wavelet_name, ExecSpace instance,
    std::index_sequence<Is...>) {
  [[maybe_unused]] bool unused =
      (distributed_transform_in<DDims, true>(
           local_grid, full_strided_domain, level, minimum_level, maximum_level,
           wavelet_name, static_cast<int>(Is), instance) &&
       ...);
}

template <typename ChunkSpanType, typename ExecSpace, typename... DDims,
          std::size_t... Is>
void dehierarchize_all_dims(
    ChunkSpanType const local_grid,
    ddc::StridedDiscreteDomain<DDims...> const &full_strided_domain,
    ddc::DiscreteVector<DDims...> const &level,
    ddc::DiscreteVector<DDims...> const &minimum_level,
    ddc::DiscreteVector<DDims...> const &maximum_level,
    std::string const &wavelet_name, ExecSpace instance,
    std::index_sequence<Is...>) {
  [[maybe_unused]] bool unused =
      (distributed_transform_in<DDims, false>(
           local_grid, full_strided_domain, level, minimum_level, maximum_level,
           wavelet_name, static_cast<int>(Is), instance) &&
       ...);
}

} // namespace detail

template <typename ChunkSpanType,
          typename ExecSpace = Kokkos::DefaultExecutionSpace, typename... DDims>
void hierarchize(
    ChunkSpanType local_grid,
    ddc::StridedDiscreteDomain<DDims...> const &full_strided_domain,
    ddc::DiscreteVector<DDims...> const &level,
    ddc::DiscreteVector<DDims...> const &minimum_level,
    ddc::DiscreteVector<DDims...> const &maximum_level,
    std::string const &wavelet_name = "hat", ExecSpace instance = ExecSpace()) {
  auto adapted = adapter_in<ExecSpace>(local_grid);

  detail::hierarchize_all_dims(adapted.view(), full_strided_domain, level,
                               minimum_level, maximum_level, wavelet_name,
                               instance, std::index_sequence_for<DDims...>{});

  adapter_out(local_grid, adapted, instance);
}

/**
 * @brief Dehierarchize across all dimensions, with ghost exchange.
 */
template <typename ChunkSpanType,
          typename ExecSpace = Kokkos::DefaultExecutionSpace, typename... DDims>
void dehierarchize(
    ChunkSpanType local_grid,
    ddc::StridedDiscreteDomain<DDims...> const &full_strided_domain,
    ddc::DiscreteVector<DDims...> const &level,
    ddc::DiscreteVector<DDims...> const &minimum_level,
    ddc::DiscreteVector<DDims...> const &maximum_level,
    std::string const &wavelet_name = "hat", ExecSpace instance = ExecSpace()) {
  auto adapted = adapter_in<ExecSpace>(local_grid);

  detail::dehierarchize_all_dims(adapted.view(), full_strided_domain, level,
                                 minimum_level, maximum_level, wavelet_name,
                                 instance, std::index_sequence_for<DDims...>{});

  adapter_out(local_grid, adapted, instance);
}

} // namespace paliwa