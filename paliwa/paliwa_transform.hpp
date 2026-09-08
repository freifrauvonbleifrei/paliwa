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

namespace detail {

// Hoisted out of transform_in(): nvcc forbids a class type local to a
// function from being used as a template argument captured by an
// extended __device__/__host__ __device__ lambda (which device_params
// is, via Kokkos::View<LevelOffsetParams*, ...>). Must live at
// namespace scope. Templated on the 1D vector/element types so each
// instantiation of transform_in still gets its own concrete type.
template <typename DVect1D, typename DElem1D>
struct LevelOffsetParams {
  int offset;
  double filter0, filter1, filter2;
  DVect1D this_d_stride;
  DVect1D virtual_length;
  DVect1D line_stride;
  DElem1D line_front;
  DElem1D write_back_1d;
  DElem1D operating_front_1d;
  long int line_extent;
};

} // namespace detail

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
  constexpr size_t dimensionality = sizeof...(DDims);

  using DElem1D = ddc::DiscreteElement<DDimInWhichToTransform>;
  using DVect1D = ddc::DiscreteVector<DDimInWhichToTransform>;
  using value_type = std::decay_t<decltype(strided_grid(chunk_domain.front()))>;

  // Namespace-scope struct, instantiated here for this function's
  // concrete DVect1D/DElem1D types. Safe to name in a KOKKOS_LAMBDA
  // capture on GPU because it is no longer a local type.
  using LevelOffsetParams = detail::LevelOffsetParams<DVect1D, DElem1D>;

  auto [even_domain_functor, odd_domain_functor] =
      get_even_and_odd_half_domain_functors<DDimInWhichToTransform, DDims...>();

  auto const initial_operating_domain = strided_domain_from_level<DDims...>(
      ddc::detail::array(level), ddc::detail::array(maximum_level));

  ddc::DiscreteVector<DDims...> pole_extents =
      initial_operating_domain.extents();
  pole_extents.template get<DDimInWhichToTransform>() = 1;
  SDDom const pole_domain(initial_operating_domain.front(), pole_extents,
                           initial_operating_domain.strides());

  std::array<long int, dimensionality> const pole_extents_arr =
      ddc::detail::array(pole_extents);
  std::array<long int, dimensionality> const pole_strides_arr =
      ddc::detail::array(initial_operating_domain.strides());
  DElem const pole_front = pole_domain.front();
  long int const n_poles = static_cast<long int>(pole_domain.size());

  auto pole_elem_from_index = KOKKOS_LAMBDA(long int idx)->DElem {
    std::array<long int, dimensionality> multi{};
    for (size_t d = 0; d < dimensionality; ++d) {
      multi[d] = (idx % pole_extents_arr[d]) * pole_strides_arr[d];
      idx /= pole_extents_arr[d];
    }
    ddc::DiscreteVector<DDims...> const offset(multi);
    return pole_front + offset;
  };

  // ---------------------------------------------------------------
  // Precompute, once on the host, everything invariant across poles
  // for each (level, offset) pair, in the original level-then-offset
  // nesting order.
  // ---------------------------------------------------------------
  long int const n_filters =
      static_cast<long int>(lifting_offsets_and_coefficients.size());

  std::vector<LevelOffsetParams> host_params;
  ddc::DiscreteVector<DDims...> current_level(level);

  for (long int current_1d_level : one_d_level_range) {
    assert(current_1d_level > 0);
    current_level.template get<DDimInWhichToTransform>() = current_1d_level;

    auto const operating_domain = strided_domain_from_level<DDims...>(
        ddc::detail::array(current_level), ddc::detail::array(maximum_level));
    auto const current_stride =
        operating_domain.strides().template get<DDimInWhichToTransform>();
    DVect1D const virtual_length(
        operating_domain.extents().template get<DDimInWhichToTransform>() *
        current_stride);
    DVect1D const this_d_stride(current_stride);
    auto const operating_front_1d =
        ddc::select<DDimInWhichToTransform>(operating_domain.front());

    for (auto const &[offset, filter] : lifting_offsets_and_coefficients) {
      std::function<SDDom(SDDom const &)> coarsen_domain;
      if (offset == 0)
        coarsen_domain = even_domain_functor;
      else if (offset == 1)
        coarsen_domain = odd_domain_functor;
      else
        throw std::runtime_error("Filter offset not supported");

      auto const write_to_domain = coarsen_domain(operating_domain);
      auto const line_front =
          ddc::select<DDimInWhichToTransform>(write_to_domain.front());
      long int const line_extent = static_cast<long int>(
          write_to_domain.extents().template get<DDimInWhichToTransform>());
      DVect1D const line_stride(
          write_to_domain.strides().template get<DDimInWhichToTransform>());
      auto const write_back_1d =
          ddc::select<DDimInWhichToTransform>(write_to_domain.back());

      // Register-carry / peeled boundary handling below assumes
      // stepping one line element forward moves exactly
      // 2 * this_d_stride, i.e. upper_element(i) == lower_element(i+1).
      // True for the standard even/odd lifting split.
      assert(line_stride == DVect1D(2 * current_stride));

      host_params.push_back(LevelOffsetParams{
          offset, filter[0], filter[1], filter[2], this_d_stride,
          virtual_length, line_stride, line_front, write_back_1d,
          operating_front_1d, line_extent});
    }
  }

  long int const n_levels =
      static_cast<long int>(host_params.size()) / n_filters;

  // WithoutInitializing: LevelOffsetParams members are fully
  // overwritten by the deep_copy below, so we skip default
  // construction on the device (also avoids requiring a
  // device-annotated default ctor for DVect1D/DElem1D).
  Kokkos::View<LevelOffsetParams *, typename ExecSpace::memory_space>
      device_params(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                        "transform_in_level_offset_params"),
                     host_params.size());
  auto host_view = Kokkos::create_mirror_view(device_params);
  for (size_t i = 0; i < host_params.size(); ++i) {
    host_view(i) = host_params[i];
  }
  Kokkos::deep_copy(device_params, host_view);

  using RangePolicy = Kokkos::RangePolicy<ExecSpace>;
  RangePolicy const policy(instance, 0, n_poles);

  Kokkos::parallel_for(
    policy, KOKKOS_LAMBDA(long int pole_idx) {
      DElem const pole_elem = pole_elem_from_index(pole_idx);

      long int idx = 0;
      for (long int level_idx = 0; level_idx < n_levels; ++level_idx) {
        for (long int offset_idx = 0; offset_idx < n_filters;
             ++offset_idx, ++idx) {
          LevelOffsetParams const &p = device_params(idx);
          value_type const f0 = static_cast<value_type>(p.filter0);
          value_type const f1 = static_cast<value_type>(p.filter1);
          value_type const f2 = static_cast<value_type>(p.filter2);

          DElem1D line_elem = p.line_front;
          bool carry_valid = false;
          value_type left{};

          for (long int line_idx = 0; line_idx < p.line_extent;
               ++line_idx, line_elem += p.line_stride) {
            DElem const ixyz =
                replace_dim<DDimInWhichToTransform>(pole_elem, line_elem);
            DElem right_elem = ixyz + p.this_d_stride;

            if ((p.offset == 1) && (line_idx == p.line_extent - 1) &&
                (DElem1D(ixyz) + p.this_d_stride > p.write_back_1d)) {
              right_elem -= p.virtual_length;
            }

            // TODO: contains() checks kept for correctness on
            // partial/sparse-domain chunk_domains (see
            // get_required_transform_domains_1d). Cannot be dropped.
            if (!chunk_domain.contains(ixyz)) {
              carry_valid = false;
              continue;
            }

            DElem lower_element = ixyz - p.this_d_stride;
            if ((p.offset == 0) && (line_idx == 0) &&
                (DElem1D(ixyz) <= p.operating_front_1d)) {
              lower_element += p.virtual_length;
            }

            if (!chunk_domain.contains(right_elem) ||
                (!carry_valid && !chunk_domain.contains(lower_element))) {
              carry_valid = false;
              continue;
            }

            value_type const lower_val =
                carry_valid ? left
                            : static_cast<value_type>(strided_grid(lower_element));
            value_type const middle = strided_grid(ixyz);
            value_type const right = strided_grid(right_elem);

            strided_grid(ixyz) = f0 * lower_val + f1 * middle + f2 * right;

            left = right;
            carry_valid = true;
          }
        }
      }
    });

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
#ifndef NDEBUG
        if (!local_grid.domain().contains(elem))
          return;
        if (!extended_span.domain().contains(elem))
          return;
#endif
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
#ifndef NDEBUG
        if (!local_grid.domain().contains(elem))
          return;
        if (!extended_span.domain().contains(elem))
          return;
#endif
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
