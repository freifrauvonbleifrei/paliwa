// SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once

#include <vector>

#include <ddc/ddc.hpp>

namespace paliwa {

template <typename... DDims>
constexpr ddc::StridedDiscreteDomain<DDims...> strided_domain_from_level(
    std::array<long int, sizeof...(DDims)> const &level,
    std::array<long int, sizeof...(DDims)> const &finest_level,
    ddc::DiscreteElement<DDims...> lbound = ddc::DiscreteElement<DDims...>()) {
  constexpr size_t dimensionality = sizeof...(DDims);
  std::array<long int, dimensionality> resolution, level_diff, stride;
  std::ranges::transform(level, resolution.begin(),
                         [](long int l) { return (1 << l); });
  std::ranges::transform(level, finest_level, level_diff.begin(),
                         [](long int l, long int ml) { return ml - l; });
  std::ranges::transform(level_diff, stride.begin(),
                         [](long int l) { return (1 << l); });
  ddc::DiscreteVector<DDims...> strides_all(stride);
  return ddc::StridedDiscreteDomain<DDims...>(
      lbound, ddc::DiscreteVector<DDims...>(resolution), strides_all);
}

template <typename... DDims>
constexpr ddc::StridedDiscreteDomain<DDims...>
strided_hierarchical_domain_from_level(
    std::array<long int, sizeof...(DDims)> const &level,
    std::array<long int, sizeof...(DDims)> const &finest_level,
    ddc::DiscreteElement<DDims...> lbound = ddc::DiscreteElement<DDims...>()) {
  constexpr size_t dimensionality = sizeof...(DDims);
  std::array<long int, dimensionality> resolution, level_diff, half_stride;
  std::ranges::transform(level, resolution.begin(),
                         [](long int ml) { return (1 << (ml - 1)); });
  std::ranges::transform(level, finest_level, level_diff.begin(),
                         [](long int l, long int ml) { return ml - l; });
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
  ddc::DiscreteVector<DDims...> resolution_all(resolution);
  ddc::DiscreteVector<DDims...> half_stride_vect(half_stride);
  ddc::DiscreteElement<DDims...> start_all = lbound + half_stride_vect;
  std::array<long int, dimensionality> stride;
  std::ranges::transform(level_diff, stride.begin(),
                         [](long int l) { return (1 << (l + 1)); });
  ddc::DiscreteVector<DDims...> strides_all(stride);
  return ddc::StridedDiscreteDomain<DDims...>(start_all, resolution_all,
                                              strides_all);
}

template <typename DDimInWhichItsOdd, typename... DDims>
constexpr ddc::StridedDiscreteDomain<DDims...> odd_strided_domain_from_domain(
    ddc::StridedDiscreteDomain<DDims...> const &domain,
    ddc::DiscreteElement<DDims...> lbound = ddc::DiscreteElement<DDims...>()) {
  ddc::DiscreteVector<DDimInWhichItsOdd> odd_offset(
      domain.strides().template get<DDimInWhichItsOdd>());
  ddc::DiscreteVector<DDims...> strides_odd = domain.strides();
  strides_odd.template get<DDimInWhichItsOdd>() *= 2;
  auto extent_odd = domain.extents();
  extent_odd.template get<DDimInWhichItsOdd>() =
      (extent_odd.template get<DDimInWhichItsOdd>()) / 2;
  return ddc::StridedDiscreteDomain<DDims...>(lbound + odd_offset, extent_odd,
                                              strides_odd);
}

template <typename DDimInWhichItsEven, typename... DDims>
constexpr ddc::StridedDiscreteDomain<DDims...> even_strided_domain_from_domain(
    ddc::StridedDiscreteDomain<DDims...> const &domain,
    ddc::DiscreteElement<DDims...> lbound = ddc::DiscreteElement<DDims...>()) {
  ddc::DiscreteVector<DDims...> strides_even = domain.strides();
  strides_even.template get<DDimInWhichItsEven>() *= 2;
  auto extent_even = domain.extents();
  extent_even.template get<DDimInWhichItsEven>() =
      (extent_even.template get<DDimInWhichItsEven>()) / 2;
  return ddc::StridedDiscreteDomain<DDims...>(lbound, extent_even,
                                              strides_even);
}

template <typename... DDims>
constexpr std::vector<ddc::StridedDiscreteDomain<DDims...>> get_strided_domains(
    std::vector<std::array<long int, sizeof...(DDims)>> const &all_levels,
    std::array<long int, sizeof...(DDims)> maximum_level,
    ddc::DiscreteElement<DDims...> lbound = ddc::DiscreteElement<DDims...>()) {
  std::vector<ddc::StridedDiscreteDomain<DDims...>> component_grid_domains;
  for (size_t grid_index = 0; grid_index < all_levels.size(); ++grid_index) {
    auto &level = all_levels[grid_index];
    component_grid_domains.emplace_back(
        strided_domain_from_level(level, maximum_level, lbound));
    // component_grid_domains.emplace_back(restrict_strided_with_discrete(
    //     strided_domain_from_level(level, maximum_level, lbound),
    //     local_domain)); #TODO
  }
  return component_grid_domains;
}

template <typename DDimInWhichToTransform, typename... DDims>
constexpr decltype(auto) get_even_and_odd_half_domain_functors() {
  return std::make_pair(
      std::bind(
          even_strided_domain_from_domain<DDimInWhichToTransform, DDims...>,
          std::placeholders::_1, ddc::DiscreteElement<DDims...>({})),
      std::bind(
          odd_strided_domain_from_domain<DDimInWhichToTransform, DDims...>,
          std::placeholders::_1, ddc::DiscreteElement<DDims...>({})));
}

template <typename OtherElementType, typename HeadTag, typename... Tags>
constexpr ddc::DiscreteElement<HeadTag, Tags...> get_intersected_begin(
    ddc::DiscreteElement<HeadTag, Tags...> const &strided_begin,
    ddc::DiscreteVector<HeadTag, Tags...> const &strides,
    OtherElementType const &other_begin) {
  // get the first element in strided that's >= other_begin
  auto const stride = ddc::select<HeadTag>(strides);
  auto const strided_begin_head = ddc::select<HeadTag>(strided_begin);
  auto const other_or_same_begin = ddc::select_or<HeadTag>(
      other_begin,
      strided_begin_head); // in case other_begin does not have this dim
  ddc::DiscreteElement<HeadTag> new_head_begin =
      strided_begin_head +
      ddc::DiscreteVector<HeadTag>(
          std::ceil((other_or_same_begin - strided_begin_head) /
                    static_cast<float>(stride)) *
          stride);
  if constexpr (sizeof...(Tags) == 0) {
    return new_head_begin;
  } else {
    return ddc::DiscreteElement<HeadTag, Tags...>(
        new_head_begin, get_intersected_begin<OtherElementType, Tags...>(
                            ddc::select<Tags...>(strided_begin),
                            ddc::select<Tags...>(strides), other_begin));
  }
}

template <class HeadTag, class... Tags, class... OtherTags>
constexpr ddc::DiscreteVector<HeadTag, Tags...> get_intersected_extent(
    ddc::DiscreteElement<HeadTag, Tags...> const &strided_begin,
    ddc::DiscreteVector<HeadTag, Tags...> const &strides,
    ddc::DiscreteVector<HeadTag, Tags...> const &strided_extents,
    ddc::DiscreteElement<OtherTags...> const &other_back) {
  ddc::DiscreteVector<HeadTag> head_result;
  if constexpr (ddc::in_tags_v<HeadTag, ddc::detail::TypeSeq<OtherTags...>>) {
    ddc::detail::array(head_result) = {(ddc::select<HeadTag>(other_back) -
                                        ddc::select<HeadTag>(strided_begin)) /
                                           strides.template get<HeadTag>() +
                                       1};
  } else {
    head_result = ddc::select<HeadTag>(strided_extents);
  }
  if constexpr (sizeof...(Tags) == 0) {
    return head_result;
  } else {
    return ddc::DiscreteVector<HeadTag, Tags...>(
        head_result,
        get_intersected_extent(ddc::select<Tags...>(strided_begin),
                               ddc::select<Tags...>(strides),
                               ddc::select<Tags...>(strided_extents),
                               ddc::select<OtherTags...>(other_back)));
  }
}

template <typename DDom, typename... DDims>
constexpr auto restrict_strided_with_discrete(
    ddc::StridedDiscreteDomain<DDims...> const &thisdomain,
    DDom const &odomain) {
  // return new strided domain that is the intersection of thisdomain and
  // odomain similar to
  // https://github.com/CExA-project/ddc/blob/d60eec09/include/ddc/discrete_domain.hpp#L197
  // KOKKOS_ASSERT(((DiscreteElement<ODDims>(m_element_begin) <=
  //                 DiscreteElement<ODDims>(odomain.m_element_begin)) &&
  //                ...))
  // KOKKOS_ASSERT(((DiscreteElement<ODDims>(m_element_end) >=
  //                 DiscreteElement<ODDims>(odomain.m_element_end)) &&
  //                ...))
  ddc::DiscreteElement<DDims...> newbegin = get_intersected_begin(
      thisdomain.front(), thisdomain.strides(), odomain.front());
  auto newextents = get_intersected_extent(
      newbegin, thisdomain.strides(), thisdomain.extents(), odomain.back());
  return ddc::StridedDiscreteDomain<DDims...>(newbegin, newextents,
                                              thisdomain.strides());
}

template <typename DDim, typename DomainType>
constexpr auto restrict_sparse_with_other_domain(
    ddc::SparseDiscreteDomain<DDim> const &sparse_domain,
    DomainType const &other_domain) {
  using DElem = ddc::DiscreteElement<DDim>;
  Kokkos::View<DElem *, Kokkos::SharedSpace> elements(
      "restricted_sparse_elements", sparse_domain.size());
  size_t insert_index = 0;
  ddc::host_for_each(sparse_domain,
                     [&elements, &insert_index, &other_domain](DElem ixyz) {
                       if (other_domain.contains(ixyz)) {
                         elements(insert_index++) = ixyz;
                       }
                     });
  Kokkos::resize(elements, insert_index);
  return ddc::SparseDiscreteDomain<DDim>(elements);
}

template <typename... DDims, typename DomainType,
          typename = std::enable_if_t<sizeof...(DDims) >= 2, int>>
constexpr auto restrict_sparse_with_other_domain(
    ddc::SparseDiscreteDomain<DDims...> const &sparse_domain,
    DomainType const &other_domain) {
  return ddc::SparseDiscreteDomain<DDims...>(restrict_sparse_with_other_domain(
      ddc::select<DDims>(sparse_domain), other_domain)...);
}

template <typename DDim>
constexpr auto sparse_from_strided_domain(
    ddc::StridedDiscreteDomain<DDim> const &strided_domain) {
  using DElem = ddc::DiscreteElement<DDim>;
  Kokkos::View<DElem *, Kokkos::SharedSpace> elements("sparse_elements",
                                                      strided_domain.size());
  size_t insert_index = 0;
  ddc::host_for_each(strided_domain, [&elements, &insert_index](DElem ixyz) {
    elements(insert_index++) = ixyz;
  });
  return ddc::SparseDiscreteDomain<DDim>(elements);
}

template <typename... DDims>
constexpr auto sparse_from_strided_domain(
    ddc::StridedDiscreteDomain<DDims...> const &strided_domain) {
  return ddc::SparseDiscreteDomain<DDims...>(
      sparse_from_strided_domain(ddc::select<DDims>(strided_domain))...);
}

template <typename DDim>
constexpr auto union_of_sparse_domains(
    ddc::SparseDiscreteDomain<DDim> const &first_sparse_domain,
    ddc::SparseDiscreteDomain<DDim> const &second_sparse_domain) {
  using DElem = ddc::DiscreteElement<DDim>;
  Kokkos::View<DElem *, Kokkos::SharedSpace> elements(
      "union_sparse_elements",
      first_sparse_domain.size() + second_sparse_domain.size());
  size_t insert_index = 0;
  auto first_domain_iterator = first_sparse_domain.begin();
  auto second_domain_iterator = second_sparse_domain.begin();
  while (first_domain_iterator != first_sparse_domain.end() &&
         second_domain_iterator != second_sparse_domain.end()) {
    if (*first_domain_iterator < *second_domain_iterator) {
      elements(insert_index++) = DElem(*first_domain_iterator);
      ++first_domain_iterator;
    } else if (*second_domain_iterator < *first_domain_iterator) {
      elements(insert_index++) = DElem(*second_domain_iterator);
      ++second_domain_iterator;
    } else {
      elements(insert_index++) = DElem(*first_domain_iterator);
      ++first_domain_iterator;
      ++second_domain_iterator;
    }
  }
  while (first_domain_iterator != first_sparse_domain.end()) {
    elements(insert_index++) = DElem(*first_domain_iterator);
    ++first_domain_iterator;
  }
  while (second_domain_iterator != second_sparse_domain.end()) {
    elements(insert_index++) = DElem(*second_domain_iterator);
    ++second_domain_iterator;
  }
  Kokkos::resize(elements, insert_index);
  return ddc::SparseDiscreteDomain<DDim>(elements);
}

template <typename... DDims,
          typename = std::enable_if_t<sizeof...(DDims) >= 2, int>>
ddc::SparseDiscreteDomain<DDims...> union_of_sparse_domains(
    ddc::SparseDiscreteDomain<DDims...> const &first_sparse_domain,
    ddc::SparseDiscreteDomain<DDims...> const &second_sparse_domain) {
  return ddc::SparseDiscreteDomain<DDims...>(
      union_of_sparse_domains(ddc::select<DDims>(first_sparse_domain),
                              ddc::select<DDims>(second_sparse_domain))...);
}

} // namespace paliwa