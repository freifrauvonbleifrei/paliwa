// SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include "paliwa_wavelets.hpp"
#include <algorithm>
#include <array>
#include <ddc/ddc.hpp>
#include <limits>
#include <map>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

namespace paliwa {

template <typename DimToReplace, typename... DDims>
constexpr auto replace_dim(ddc::DiscreteElement<DDims...> base,
                           ddc::DiscreteElement<DimToReplace> d_elem) {
  /// Replace one dimension's component in a multi-D DiscreteElement
  return ddc::DiscreteElement<DDims...>([&]() -> ddc::DiscreteElement<DDims> {
    if constexpr (std::is_same_v<DDims, DimToReplace>) {
      return ddc::DiscreteElement<DDims>(d_elem);
    } else {
      return ddc::select<DDims>(base);
    }
  }()...);
}

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
constexpr ddc::SparseDiscreteDomain<DDim> restrict_sparse_with_other_domain(
    ddc::SparseDiscreteDomain<DDim> const &sparse_domain,
    DomainType const &other_domain) {
  using DElem = ddc::DiscreteElement<DDim>;
  Kokkos::View<DElem *, Kokkos::SharedSpace> elements(
      "restricted_sparse_elements", sparse_domain.size());
  auto other_domain_projected = ddc::select<DDim>(other_domain);
  size_t insert_index = 0;
  ddc::host_for_each(sparse_domain, [&elements, &insert_index,
                                     &other_domain_projected](DElem ixyz) {
    if (other_domain_projected.contains(ixyz)) {
      elements(insert_index++) = DElem(ixyz);
    }
  });
  Kokkos::resize(elements, insert_index);
  return ddc::SparseDiscreteDomain<DDim>(elements);
}

template <typename FirstDim, typename... DDims, typename DomainType,
          typename = std::enable_if_t<sizeof...(DDims) >= 1, int>>
constexpr ddc::SparseDiscreteDomain<FirstDim, DDims...>
restrict_sparse_with_other_domain(
    ddc::SparseDiscreteDomain<FirstDim, DDims...> const &sparse_domain,
    DomainType const &other_domain) {
  return ddc::SparseDiscreteDomain<FirstDim, DDims...>(
      restrict_sparse_with_other_domain<FirstDim>(
          ddc::SparseDiscreteDomain<FirstDim>(sparse_domain), other_domain),
      restrict_sparse_with_other_domain<DDims...>(
          ddc::SparseDiscreteDomain<DDims...>(sparse_domain), other_domain));
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
  return ddc::SparseDiscreteDomain<DDims...>(union_of_sparse_domains(
      ddc::SparseDiscreteDomain<DDims>(first_sparse_domain),
      ddc::SparseDiscreteDomain<DDims>(second_sparse_domain))...);
}

// Inclusive periodic bounds in finest-level coordinates. min > max wraps;
// {-1, -1} is empty. The stride is supplied by the hierarchical level/pass.
struct PeriodicBounds {
  long min = -1;
  long max = -1;
  bool empty() const { return min < 0; }
};

namespace detail {
// Work on an unwrapped axis so expansion across zero needs no special cases.
struct LevelInterval {
  long min = 1;
  long max = 0;
  bool empty() const { return min > max; }

  LevelInterval aligned(long stride, long offset) const {
    if (empty())
      return {};
    auto mod = [stride](long x) { return (x % stride + stride) % stride; };
    return {min + mod(offset - min), max - mod(max - offset)};
  }

  void include(LevelInterval other) {
    if (other.empty())
      return;
    if (empty())
      *this = other;
    else {
      min = std::min(min, other.min);
      max = std::max(max, other.max);
    }
  }

  PeriodicBounds periodic(long period, long stride, long offset) const {
    if (empty())
      return {};
    // Canonical full interval on this lattice; never enumerate a point twice.
    if ((max - min) / stride + 1 >= period / stride)
      return {offset, period - stride + offset};
    auto mod = [period](long x) { return (x % period + period) % period; };
    return {mod(min), mod(max)};
  }
};
} // namespace detail

// Dependency state: one interval for each hierarchical level's odd points;
// level zero holds the periodic coarse point. This partitions the grid, so
// coarse even-only requirements never create spurious finer-level odd points.
// Visit writes backwards; callers may save the small pass bounds to execute
// forwards. No coefficient values are inspected: every stencil reads all taps.
template <typename LevelRange, typename VisitPass>
std::vector<PeriodicBounds> transform_bounds(
    PeriodicBounds outputs, long level, long maximum_level,
    LevelRange const &levels,
    std::vector<std::pair<int, std::array<double, 3>>> const &filters,
    VisitPass visit_pass) {
  // Consecutive levels preserve the interval invariant. Skipping a level can
  // leave holes among its even points and requires a more general set planner.
  std::vector<long> ordered_levels(levels.begin(), levels.end());
  if (ordered_levels.size() > 1) {
    long const step = ordered_levels[1] - ordered_levels[0];
    if (step != 1 && step != -1)
      throw std::invalid_argument("Transform levels must be consecutive");
    for (size_t i = 1; i < ordered_levels.size(); ++i)
      if (ordered_levels[i] - ordered_levels[i - 1] != step)
        throw std::invalid_argument(
            "Transform levels must be monotone and consecutive");
  }
  long const period = 1L << maximum_level;
  std::vector<detail::LevelInterval> required(level + 1);
  auto lattice_stride = [&](long l) {
    return l == 0 ? period : 2L << (maximum_level - l);
  };
  auto lattice_offset = [&](long l) {
    return l == 0 ? 0L : 1L << (maximum_level - l);
  };
  detail::LevelInterval output_interval;
  if (!outputs.empty()) {
    output_interval = {outputs.min,
                       outputs.max + (outputs.min > outputs.max ? period : 0)};
  }
  for (long l = 0; l <= level; ++l)
    required[l] = output_interval.aligned(lattice_stride(l), lattice_offset(l));

  for (size_t i = ordered_levels.size(); i-- > 0;) {
    long const l = ordered_levels[i];
    if (l < 1 || l > level)
      throw std::invalid_argument("Invalid transform level");
    long const stride = 1L << (maximum_level - l);
    for (size_t pass = filters.size(); pass-- > 0;) {
      int const offset = filters[pass].first;
      if (offset != 0 && offset != 1)
        throw std::invalid_argument("Filter offset not supported");
      auto writes = required[l];
      if (offset == 0) {
        writes = {};
        // Even points at this level are precisely the coarser levels.
        for (long h = 0; h < l; ++h)
          writes.include(required[h]);
      }
      visit_pass(i, pass, writes.periodic(period, 2 * stride, offset * stride));
      if (writes.empty())
        continue;
      detail::LevelInterval const neighbors{writes.min - stride,
                                            writes.max + stride};
      if (offset == 0) {
        required[l].include(neighbors.aligned(2 * stride, stride));
      } else {
        for (long h = 0; h < l; ++h)
          required[h].include(
              neighbors.aligned(lattice_stride(h), lattice_offset(h)));
      }
    }
  }
  std::vector<PeriodicBounds> bounds(level + 1);
  for (long l = 0; l <= level; ++l)
    bounds[l] =
        required[l].periodic(period, lattice_stride(l), lattice_offset(l));
  return bounds;
}

// Visit the disjoint level intervals without allocating sparse storage.
template <typename Dim, typename Visit>
void for_each_transform_index(std::vector<PeriodicBounds> const &bounds,
                              long maximum_level, Visit visit) {
  long const period = 1L << maximum_level;
  for (size_t l = 0; l < bounds.size(); ++l) {
    auto const b = bounds[l];
    if (b.empty())
      continue;
    long const stride = l == 0 ? period : 2L << (maximum_level - l);
    long const last = b.max + (b.min > b.max ? period : 0);
    for (long x = b.min; x <= last; x += stride)
      visit(ddc::DiscreteElement<Dim>(x % period));
  }
}

// Materialize the disjoint level intervals only where sparse storage is needed.
template <typename Dim>
auto domain_from_transform_bounds(std::vector<PeriodicBounds> const &bounds,
                                  long maximum_level) {
  using Element = ddc::DiscreteElement<Dim>;
  size_t count = 0;
  for_each_transform_index<Dim>(bounds, maximum_level,
                                [&](Element) { ++count; });
  Kokkos::View<Element *, Kokkos::SharedSpace> elements("required_elements",
                                                        count);
  size_t i = 0;
  for_each_transform_index<Dim>(bounds, maximum_level,
                                [&](Element x) { elements(i++) = x; });
  if (count > 1)
    std::sort(elements.data(), elements.data() + count);
  return ddc::SparseDiscreteDomain<Dim>(elements);
}

// Smallest circular interval containing sorted, distinct indices
inline PeriodicBounds covering_periodic_bounds(std::vector<long> const &indices,
                                               long period) {
  if (indices.empty())
    return {};
  PeriodicBounds result{indices.front(), indices.back()};
  long largest_gap = indices.front() + period - indices.back();
  for (size_t i = 1; i < indices.size(); ++i) {
    long const gap = indices[i] - indices[i - 1];
    if (gap > largest_gap) {
      largest_gap = gap;
      result = {indices[i], indices[i - 1]};
    }
  }
  return result;
}

// One affine piece of a lifting pass in packed-pole coordinates. Splitting
// at the periodic edges keeps wrap checks out of the per-point loop.
struct PolePassSegment {
  int first, count, stride;
  int lower, upper;
};

inline std::vector<PolePassSegment>
pole_pass_segments(PeriodicBounds writes, ddc::DiscreteElementType origin,
                   int level, int maximum_level, int pass_level) {
  using Coordinate = ddc::DiscreteElementType;
  std::vector<PolePassSegment> result;
  if (writes.empty())
    return result;
  if (level < 1 || level >= std::numeric_limits<int>::digits ||
      maximum_level < level ||
      maximum_level >= std::numeric_limits<Coordinate>::digits ||
      pass_level < 1 || pass_level > level)
    throw std::invalid_argument("Invalid levels for packed-pole indexing");
  auto const period = 1 << level;
  auto const distance = 1 << (level - pass_level);
  auto const step = 2 * distance;
  auto const spacing_log2 = maximum_level - level;
  auto const global_period = Coordinate{1} << maximum_level;
  // Keep global coordinates wide; only packed-pole indices need to fit int.
  auto const first = static_cast<Coordinate>(writes.min);
  auto const last = static_cast<Coordinate>(writes.max);
  auto const span =
      last >= first ? last - first : global_period - (first - last);
  auto remaining = static_cast<int>(span >> spacing_log2) / step + 1;
  auto x = static_cast<int>(
      (first >= origin ? first - origin : global_period - (origin - first)) >>
      spacing_log2);
  while (remaining > 0) {
    auto const lower = x < distance ? period - (distance - x) : x - distance;
    auto const upper =
        x >= period - distance ? x - (period - distance) : x + distance;
    auto const end =
        x < distance
            ? distance - 1
            : (x < period - distance ? period - distance - 1 : period - 1);
    auto const count = std::min(remaining, (end - x) / step + 1);
    result.push_back({x, count, step, lower, upper});
    remaining -= count;
    auto const advance = count * step;
    x = advance >= period - x ? advance - (period - x) : x + advance;
  }
  return result;
}

// Intersect one axis of a strided domain with a periodic interval. A wrapped
// interval becomes two ordinary domains; alignment always uses global parity.
template <typename Dim, typename... Dims>
std::vector<ddc::StridedDiscreteDomain<Dims...>>
restrict_periodic_bounds(ddc::StridedDiscreteDomain<Dims...> const &domain,
                         PeriodicBounds bounds, long period) {
  std::vector<ddc::StridedDiscreteDomain<Dims...>> result;
  if (bounds.empty() || domain.empty())
    return result;
  long const first = domain.front().template uid<Dim>();
  long const last = domain.back().template uid<Dim>();
  long const stride = domain.strides().template get<Dim>();
  auto append = [&](long lower, long upper) {
    lower = std::max(lower, first);
    upper = std::min(upper, last);
    long const begin = first + ((lower - first + stride - 1) / stride) * stride;
    if (begin > upper)
      return;
    auto extents = domain.extents();
    extents.template get<Dim>() = (upper - begin) / stride + 1;
    result.emplace_back(
        replace_dim(domain.front(), ddc::DiscreteElement<Dim>(begin)), extents,
        domain.strides());
  };
  if (bounds.min <= bounds.max) {
    append(bounds.min, bounds.max);
  } else {
    append(bounds.min, period - 1);
    append(0, bounds.max);
  }
  return result;
}

// Use the global transform axis and local pole coordinates in other axes.
template <typename Dim, typename... Dims>
auto local_pole_domain(ddc::StridedDiscreteDomain<Dims...> const &global,
                       ddc::StridedDiscreteDomain<Dims...> const &local) {
  return ddc::StridedDiscreteDomain<Dims...>(
      replace_dim(local.front(), ddc::select<Dim>(global.front())),
      ddc::DiscreteVector<Dims...>([&]() {
        if constexpr (std::is_same_v<Dims, Dim>)
          return global.extents().template get<Dims>();
        else
          return local.extents().template get<Dims>();
      }()...),
      ddc::DiscreteVector<Dims...>([&]() {
        if constexpr (std::is_same_v<Dims, Dim>)
          return global.strides().template get<Dims>();
        else
          return local.strides().template get<Dims>();
      }()...));
}

template <typename SelectedDim, typename SDDom, typename DDom>
constexpr ddc::SparseDiscreteDomain<SelectedDim> get_required_transform_domain(
    bool is_for_hierarchization, SDDom const &full_domain,
    DDom const &local_domain, ddc::DiscreteVector<SelectedDim> const &level,
    ddc::DiscreteVector<SelectedDim> const &minimum_level,
    ddc::DiscreteVector<SelectedDim> const &maximum_level,
    std::string const &wavelet_name = "hat") {
  std::vector<long> levels;
  for (long l = static_cast<long>(minimum_level) + 1;
       l <= static_cast<long>(level); ++l)
    levels.push_back(l);
  if (is_for_hierarchization)
    std::ranges::reverse(levels);
  auto const &filters =
      is_for_hierarchization
          ? lifting_wavelet_filter_offsets_and_coefficients.at(wavelet_name)
          : lifting_wavelet_reconstruct_offsets_and_coefficients.at(
                wavelet_name);
  PeriodicBounds outputs;
  if (!local_domain.empty())
    outputs = {
        static_cast<long>(local_domain.front().template uid<SelectedDim>()),
        static_cast<long>(local_domain.back().template uid<SelectedDim>())};
  long const max_level = static_cast<long>(maximum_level);
  auto const bounds =
      transform_bounds(outputs, static_cast<long>(level), max_level, levels,
                       filters, [](size_t, size_t, PeriodicBounds) {});
  auto const required =
      domain_from_transform_bounds<SelectedDim>(bounds, max_level);
  using Element = ddc::DiscreteElement<SelectedDim>;
  Kokkos::View<Element *, Kokkos::SharedSpace> ghosts("ghost_elements",
                                                      required.size());
  size_t count = 0;
  ddc::host_for_each(required, [&](Element i) {
    if (!full_domain.contains(i))
      throw std::runtime_error("Required input outside full domain");
    if (!local_domain.contains(i))
      ghosts(count++) = i;
  });
  Kokkos::resize(ghosts, count);
  return ddc::SparseDiscreteDomain<SelectedDim>(ghosts);
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
