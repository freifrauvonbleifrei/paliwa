#pragma once

#include <ranges>
#include <vector>

#include <ddc/ddc.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "paliwa_domains.hpp"
#include "paliwa_wavelets.hpp"

template <typename DDimInWhichToTransform,
          typename DDomainType,   // TODO either DDom or SDDom or SparseDDom
          typename ChunkSpanType, // TODO w.r.t. DDomainType
          typename LevelRange,    // TODO input_range concept
          typename ExecSpace,     // todo = Kokkos::DefaultExecutionSpace,
          typename... DDims>
bool transform_in(DDomainType const &strided_domain,
                  ChunkSpanType const strided_grid,
                  ddc::DiscreteVector<DDims...> const &level,
                  ddc::DiscreteVector<DDims...> const &maximum_level,
                  ddc::DiscreteElement<DDims...> const &lbound,
                  LevelRange const &one_d_level_range,
                  std::vector<std::pair<int, std::array<double, 3>>> const
                      &lifting_offsets_and_coefficients,
                  ExecSpace instance = ExecSpace()) {
  using DElem = ddc::DiscreteElement<DDims...>;
  using SDDom = ddc::StridedDiscreteDomain<DDims...>;
  static_assert(
      std::is_same_v<typename DDomainType::discrete_element_type, DElem>,
      "Mismatch between DDomainType and DDims...");

  auto const ddc_level_1d_vec = ddc::select<DDimInWhichToTransform>(level);
  auto const ddc_max_level_1d_vec =
      ddc::select<DDimInWhichToTransform>(maximum_level);

  // check the finest stride, if DDomainType is SDDom
  if constexpr (std::is_same_v<DDomainType, SDDom>) {
    assert((1 << (ddc_max_level_1d_vec - ddc_level_1d_vec)) ==
           strided_domain.strides().template get<DDimInWhichToTransform>());
  }
  ddc::DiscreteVector<DDims...> current_level(level);

  for (long int current_1d_level : one_d_level_range) {
    int const current_stride = (1 << (ddc_max_level_1d_vec - current_1d_level));
    current_level.template get<DDimInWhichToTransform>() = current_1d_level;
    auto const operating_domain =
        strided_domain_from_level(ddc::detail::array(current_level),
                                  ddc::detail::array(maximum_level), lbound);

    auto virtual_length = ddc::DiscreteVector<DDimInWhichToTransform>(
        operating_domain.extents().template get<DDimInWhichToTransform>() *
        operating_domain.strides().template get<DDimInWhichToTransform>());

    for (auto const &[offset, filter] : lifting_offsets_and_coefficients) {
      // access chunk at every other point in transform dimension
      std::function<SDDom(SDDom const &)> coarsen_domain;
      if (offset == 0) {
        coarsen_domain = std::bind(
            even_strided_domain_from_domain<DDimInWhichToTransform, DDims...>,
            std::placeholders::_1, lbound);
      } else if (offset == 1) {
        coarsen_domain = std::bind(
            odd_strided_domain_from_domain<DDimInWhichToTransform, DDims...>,
            std::placeholders::_1, lbound);
      } else {
        throw std::runtime_error("Filter offset not supported");
      }
      auto const write_to_domain = coarsen_domain(operating_domain);

      ddc::parallel_for_each(
          instance, write_to_domain, KOKKOS_LAMBDA(DElem const ixyz) {
            // check for out of bounds, periodic if necessary
            auto const this_d_stride =
                ddc::DiscreteVector<DDimInWhichToTransform>(current_stride);
            if ((offset == 1) &&
                (ddc::DiscreteElement<DDimInWhichToTransform>(ixyz) +
                     this_d_stride >
                 ddc::DiscreteElement<DDimInWhichToTransform>(
                     write_to_domain.back()))) {
              // on the upper boundary, no +1 available
              // TODO make separate step to avoid branch here?
              strided_grid(ixyz) =
                  filter[0] * strided_grid(ixyz - this_d_stride) +
                  filter[1] * strided_grid(ixyz) +
                  filter[2] *
                      strided_grid(ixyz + this_d_stride - virtual_length);
            } else if ((offset == 0) &&
                       (ddc::DiscreteElement<DDimInWhichToTransform>(ixyz) <=
                        ddc::DiscreteElement<DDimInWhichToTransform>(lbound))) {
              // on the lower boundary, no -1 available
              strided_grid(ixyz) =
                  filter[0] *
                      strided_grid(ixyz - this_d_stride + virtual_length) +
                  filter[1] * strided_grid(ixyz) +
                  filter[2] * strided_grid(ixyz + this_d_stride);
            } else {
              strided_grid(ixyz) =
                  filter[0] * strided_grid(ixyz - this_d_stride) +
                  filter[1] * strided_grid(ixyz) +
                  filter[2] * strided_grid(ixyz + this_d_stride);
            }
          });
    }
  }
  return true;
}

template <typename DDimInWhichToHierarchize, typename DDomainType,
          typename ChunkSpanType,
          typename ExecSpace, // = Kokkos::DefaultExecutionSpace
          typename... DDims>
bool hierarchize_in(DDomainType const &strided_domain,
                    ChunkSpanType const strided_grid,
                    ddc::DiscreteVector<DDims...> const &level,
                    ddc::DiscreteVector<DDims...> const &minimum_level,
                    ddc::DiscreteVector<DDims...> const &maximum_level,
                    ddc::DiscreteElement<DDims...> const &lbound,
                    std::string const &wavelet_name = "hat",
                    ExecSpace instance = ExecSpace()) {

  auto const ddc_level_1d_vec = ddc::select<DDimInWhichToHierarchize>(level);
  auto const ddc_min_level_1d_vec =
      ddc::select<DDimInWhichToHierarchize>(minimum_level);

  auto decreasing_range =
      std::views::iota(static_cast<long int>(ddc_min_level_1d_vec) - 1,
                       static_cast<long int>(ddc_level_1d_vec) + 1) |
      std::views::reverse;
  return transform_in<DDimInWhichToHierarchize>(
      strided_domain, strided_grid, level, maximum_level, lbound,
      decreasing_range,
      lifting_wavelet_filter_offsets_and_coefficients.at(wavelet_name),
      instance);
}

template <typename DDimInWhichToHierarchize, typename DDomainType,
          typename ChunkSpanType,
          typename ExecSpace, // = Kokkos::DefaultExecutionSpace
          typename... DDims>
bool dehierarchize_in(DDomainType const &strided_domain,
                      ChunkSpanType const strided_grid,
                      ddc::DiscreteVector<DDims...> const &level,
                      ddc::DiscreteVector<DDims...> const &minimum_level,
                      ddc::DiscreteVector<DDims...> const &maximum_level,
                      ddc::DiscreteElement<DDims...> const &lbound,
                      std::string const &wavelet_name = "hat",
                      ExecSpace instance = ExecSpace()) {

  auto const ddc_level_1d_vec = ddc::select<DDimInWhichToHierarchize>(level);
  auto const ddc_min_level_1d_vec =
      ddc::select<DDimInWhichToHierarchize>(minimum_level);

  auto increasing_range =
      std::views::iota(static_cast<long int>(ddc_min_level_1d_vec) - 1,
                       static_cast<long int>(ddc_level_1d_vec) + 1);
  return transform_in<DDimInWhichToHierarchize>(
      strided_domain, strided_grid, level, maximum_level, lbound,
      increasing_range,
      lifting_wavelet_reconstruct_offsets_and_coefficients.at(wavelet_name),
      instance);
}

template <typename DDomainType, typename ChunkSpanType,
          typename ExecSpace, // = Kokkos::DefaultExecutionSpace
          typename... DDims>
void hierarchize(DDomainType const &strided_domain,
                 ChunkSpanType const strided_grid,
                 ddc::DiscreteVector<DDims...> const &level,
                 ddc::DiscreteVector<DDims...> const &minimum_level,
                 ddc::DiscreteVector<DDims...> const &maximum_level,
                 ddc::DiscreteElement<DDims...> const &lbound,
                 std::string const &wavelet_name = "hat",
                 ExecSpace instance = ExecSpace()) {

  // fold expression to call for every dimension
  bool unused =
      (hierarchize_in<DDims>(strided_domain, strided_grid, level, minimum_level,
                             maximum_level, lbound, wavelet_name, instance) &&
       ...);
}

template <typename DDomainType, typename ChunkSpanType,
          typename ExecSpace, // = Kokkos::DefaultExecutionSpace
          typename... DDims>
void dehierarchize(DDomainType const &strided_domain,
                   ChunkSpanType const strided_grid,
                   ddc::DiscreteVector<DDims...> const &level,
                   ddc::DiscreteVector<DDims...> const &minimum_level,
                   ddc::DiscreteVector<DDims...> const &maximum_level,
                   ddc::DiscreteElement<DDims...> const &lbound,
                   std::string const &wavelet_name = "hat",
                   ExecSpace instance = ExecSpace()) {

  // fold expression to call for every dimension
  bool unused = (dehierarchize_in<DDims>(strided_domain, strided_grid, level,
                                         minimum_level, maximum_level, lbound,
                                         wavelet_name, instance) &&
                 ...);
}

template <typename SelectedDim, typename SDDom, typename DDom>
ddc::SparseDiscreteDomain<SelectedDim> get_required_transform_domain(
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
    dehierarchize_in<SelectedDim>(
        full_domain, full_pole, level, minimum_level, maximum_level,
        full_domain.front(), wavelet_name, Kokkos::DefaultHostExecutionSpace());
  } else {
    hierarchize_in<SelectedDim>(
        full_domain, full_pole, level, minimum_level, maximum_level,
        full_domain.front(), wavelet_name, Kokkos::DefaultHostExecutionSpace());
  }
  // set to 0.0 on local_domain
  ddc::parallel_for_each(
      Kokkos::DefaultHostExecutionSpace(), local_domain,
      KOKKOS_LAMBDA(DElem const ixyz) { full_pole(ixyz) = 0.0; });

  // extract remaining indices to domain
  Kokkos::View<DElem *, Kokkos::SharedSpace> required_elements(
      "required_elements", full_domain.size());
  size_t insert_index = 0;
  ddc::for_each(full_domain,
                [&required_elements, &full_pole, &insert_index](DElem ixyz) {
                  if (full_pole(ixyz) != 0.0) {
                    required_elements(insert_index++) = ixyz;
                  }
                });
  Kokkos::resize(required_elements, insert_index);
  return ddc::SparseDiscreteDomain<SelectedDim>(required_elements);
}

template <typename DDom, typename HeadTag, typename... DDims>
ddc::SparseDiscreteDomain<HeadTag, DDims...>
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
ddc::SparseDiscreteDomain<DDims...> get_required_transform_domains(
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