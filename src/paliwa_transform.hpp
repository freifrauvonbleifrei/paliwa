#pragma once

#include <ranges>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <ddc/ddc.hpp>

#include "paliwa_domains.hpp"
#include "paliwa_wavelets.hpp"

namespace paliwa {

template <typename DDimInWhichToTransform, typename ChunkSpanType,
          typename LevelRange, // TODO input_range concept
          typename ExecSpace,  // todo = Kokkos::DefaultExecutionSpace,
          typename... DDims>
constexpr bool
transform_in(ChunkSpanType const strided_grid,
             ddc::DiscreteVector<DDims...> const &level,
             ddc::DiscreteVector<DDims...> const &maximum_level,
             LevelRange const &one_d_level_range,
             std::vector<std::pair<int, std::array<double, 3>>> const
                 &lifting_offsets_and_coefficients,
             ExecSpace instance = ExecSpace()) {
  using DElem = ddc::DiscreteElement<DDims...>;
  using SDDom = ddc::StridedDiscreteDomain<DDims...>;
  auto [even_domain_functor, odd_domain_functor] =
      get_even_and_odd_half_domain_functors<DDimInWhichToTransform, DDims...>();

  ddc::DiscreteVector<DDims...> current_level(level);
  for (long int current_1d_level : one_d_level_range) {
    assert(current_1d_level >= 0);
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
      // access chunk at every other point in transform dimension
      std::function<SDDom(SDDom const &)> coarsen_domain;
      if (offset == 0) {
        coarsen_domain = even_domain_functor;
      } else if (offset == 1) {
        coarsen_domain = odd_domain_functor;
      } else {
        throw std::runtime_error("Filter offset not supported");
      }
      auto const write_to_domain = coarsen_domain(operating_domain);

      ddc::parallel_for_each(
          instance, write_to_domain, KOKKOS_LAMBDA(DElem const ixyz) {
            // check for out of bounds, periodic if necessary
            DElem lower_element = ixyz - this_d_stride;
            DElem upper_element = ixyz + this_d_stride;
            if ((offset == 1) &&
                (ddc::DiscreteElement<DDimInWhichToTransform>(ixyz) +
                     this_d_stride >
                 ddc::DiscreteElement<DDimInWhichToTransform>(
                     write_to_domain.back()))) {
              // on the upper boundary, no +1 available
              // TODO make separate step to avoid branch here?
              upper_element -= virtual_length;
            } else if ((offset == 0) &&
                       (ddc::DiscreteElement<DDimInWhichToTransform>(ixyz) <=
                        ddc::select<DDimInWhichToTransform>(
                            operating_domain.front()))) {
              // on the lower boundary, no -1 available
              lower_element += virtual_length;
            }
            strided_grid(ixyz) = filter[0] * strided_grid(lower_element) +
                                 filter[1] * strided_grid(ixyz) +
                                 filter[2] * strided_grid(upper_element);
          });
    }
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

template <typename SelectedDim, typename SDDom, typename DDom>
constexpr ddc::SparseDiscreteDomain<SelectedDim> get_required_transform_domain(
    bool is_for_hierarchization, SDDom const &full_domain,
    DDom const &local_domain, ddc::DiscreteVector<SelectedDim> const &level,
    ddc::DiscreteVector<SelectedDim> const &minimum_level,
    ddc::DiscreteVector<SelectedDim> const &maximum_level,
    std::string const &wavelet_name = "hat") {
  // dehierarchize a 1d pole, where all is initialized to 0 except for the
  // local domain, to get all indices that will be required for hierarchization
  using DElem = ddc::DiscreteElement<SelectedDim>;

  // initialize full pole to zero
  ddc::Chunk full_pole_chunk("full_pole_chunk", full_domain,
                             ddc::HostAllocator<double>());
  auto full_pole = full_pole_chunk.span_view();
  ddc::parallel_for_each(
      Kokkos::DefaultHostExecutionSpace(), full_domain,
      KOKKOS_LAMBDA(DElem const ixyz) { full_pole(ixyz) = 0.0; });
  // cf.
  // https://kokkos.org/kokkos-core-wiki/API/algorithms/Random-Number.html#example
  Kokkos::Random_XorShift64_Pool<Kokkos::HostSpace> random_pool(/*seed=*/12345);
  ddc::parallel_for_each(
      Kokkos::DefaultHostExecutionSpace(), local_domain,
      KOKKOS_LAMBDA(DElem const ixyz) {
        auto generator = random_pool.get_state();
        double random_number = generator.drand(0., 1.);
        random_pool.free_state(generator);
        full_pole(ixyz) = ixyz.uid() * 100000.0 *
                          (2 + random_number); // just some non-zero value
        // TODO make all stencil values non-negative instead
      });
  if (is_for_hierarchization) {
    dehierarchize_in<SelectedDim>(full_pole, level, minimum_level,
                                  maximum_level, wavelet_name,
                                  Kokkos::DefaultHostExecutionSpace());
  } else {
    hierarchize_in<SelectedDim>(full_pole, level, minimum_level, maximum_level,
                                wavelet_name,
                                Kokkos::DefaultHostExecutionSpace());
  }
  // set to 0.0 on local_domain
  ddc::parallel_for_each(
      Kokkos::DefaultHostExecutionSpace(), local_domain,
      KOKKOS_LAMBDA(DElem const ixyz) { full_pole(ixyz) = 0.0; });

  // extract remaining indices to domain
  Kokkos::View<DElem *, Kokkos::SharedSpace> required_elements(
      "required_elements", full_domain.size());
  size_t insert_index = 0;
  ddc::host_for_each(
      full_domain, [&required_elements, &full_pole, &insert_index](DElem ixyz) {
        if (full_pole(ixyz) != 0.0) {
          required_elements(insert_index++) = ixyz;
        }
      });
  Kokkos::resize(required_elements, insert_index);
  return ddc::SparseDiscreteDomain<SelectedDim>(required_elements);
}

template <typename DDom, typename HeadTag, typename... DDims>
constexpr ddc::SparseDiscreteDomain<HeadTag, DDims...>
get_required_transform_domains_recursive(
    bool is_for_hierarchization,
    ddc::StridedDiscreteDomain<HeadTag, DDims...> const &full_domain,
    DDom const &local_domain,
    ddc::DiscreteVector<HeadTag, DDims...> const &level,
    ddc::DiscreteVector<HeadTag, DDims...> const &minimum_level,
    ddc::DiscreteVector<HeadTag, DDims...> const &maximum_level,
    std::string const &wavelet_name = "hat") {
  auto head_domain = get_required_transform_domain<HeadTag>(
      is_for_hierarchization, ddc::select<HeadTag>(full_domain),
      ddc::select<HeadTag>(local_domain), ddc::select<HeadTag>(level),
      ddc::select<HeadTag>(minimum_level), ddc::select<HeadTag>(maximum_level),
      wavelet_name);
  if constexpr (sizeof...(DDims) == 0) {
    return head_domain;
  } else {
    return ddc::SparseDiscreteDomain<HeadTag, DDims...>(
        head_domain,
        get_required_transform_domains_recursive(
            is_for_hierarchization, ddc::select<DDims...>(full_domain),
            ddc::select<DDims...>(local_domain), ddc::select<DDims...>(level),
            ddc::select<DDims...>(minimum_level),
            ddc::select<DDims...>(maximum_level), wavelet_name));
  }
}

template <typename DDom, typename... DDims>
constexpr ddc::SparseDiscreteDomain<DDims...> get_required_transform_domains(
    bool is_for_hierarchization,
    ddc::StridedDiscreteDomain<DDims...> const &full_domain,
    DDom const &local_domain, ddc::DiscreteVector<DDims...> const &level,
    ddc::DiscreteVector<DDims...> const &minimum_level,
    ddc::DiscreteVector<DDims...> const &maximum_level,
    std::string const &wavelet_name = "hat") {
  return ddc::SparseDiscreteDomain<DDims...>(
      get_required_transform_domains_recursive(
          is_for_hierarchization, full_domain, local_domain, level,
          minimum_level, maximum_level, wavelet_name));
}
} // namespace paliwa