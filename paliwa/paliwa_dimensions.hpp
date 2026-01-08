// SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

namespace paliwa {
struct A {};
struct B {};
struct C {};
struct D {};
struct E {};
struct F {};
struct G {};
struct H {};
struct I {};
struct J {};
struct K {};
struct L {};
struct M {};
struct N {};
struct O {};
struct P {};
struct Q {};
struct R {};
struct S {};
struct T {};
struct U {};
struct V {};
struct W {};
struct X {};
struct Y {};
struct Z {};

template <typename T, size_t type_number> struct NumberedType : T {
  static constexpr size_t ID = type_number;
};

template <class T, size_t type_number = 0>
struct D_Dim : NumberedType<ddc::UniformPointSampling<T>, type_number> {};

using DDimA = D_Dim<A>;
using DDimB = D_Dim<B>;
using DDimC = D_Dim<C>;
using DDimD = D_Dim<D>;
using DDimE = D_Dim<E>;
using DDimF = D_Dim<F>;
using DDimG = D_Dim<G>;
using DDimH = D_Dim<H>;
using DDimI = D_Dim<I>;
using DDimJ = D_Dim<J>;
using DDimK = D_Dim<K>;
using DDimL = D_Dim<L>;
using DDimM = D_Dim<M>;
using DDimN = D_Dim<N>;
using DDimO = D_Dim<O>;
using DDimP = D_Dim<P>;
using DDimQ = D_Dim<Q>;
using DDimR = D_Dim<R>;
using DDimS = D_Dim<S>;
using DDimT = D_Dim<T>;
using DDimU = D_Dim<U>;
using DDimV = D_Dim<V>;
using DDimW = D_Dim<W>;
using DDimX = D_Dim<X>;
using DDimY = D_Dim<Y>;
using DDimZ = D_Dim<Z>;

template <typename DDim>
ddc::DiscreteDomain<DDim> initialize_dim_periodic_unit_interval(
    ddc::DiscreteVector<DDim> const &resolution) {
  using CDim = typename DDim::continuous_dimension_type;
  auto const domain_with_periodic_point =
      ddc::init_discrete_space<DDim>(DDim::template init<DDim>(
          ddc::Coordinate<CDim>(0.0), ddc::Coordinate<CDim>(1.0), resolution));
  return domain_with_periodic_point.remove_last(ddc::DiscreteVector<DDim>(1));
}

template <typename DDimToInitialize, typename... DDims,
          typename = std::enable_if_t<(sizeof...(DDims) > 1)>>
ddc::DiscreteDomain<DDimToInitialize>
optional_initialize_dims_periodic_unit_intervals(
    ddc::DiscreteVector<DDims...> const &resolution) {
  if (ddc::is_discrete_space_initialized<DDimToInitialize>()) {
    assert(ddc::host_discrete_space<DDimToInitialize>().origin() == 0.0);
    assert(ddc::host_discrete_space<DDimToInitialize>().step() ==
           1.0 / (ddc::select<DDimToInitialize>(resolution) - 1));
    assert(ddc::host_discrete_space<DDimToInitialize>().front() ==
           ddc::DiscreteElement<DDimToInitialize>(0));
    return ddc::DiscreteDomain<DDimToInitialize>();
  } else {
    return initialize_dim_periodic_unit_interval<DDimToInitialize>(
        ddc::select<DDimToInitialize>(resolution));
  }
}

template <typename... DDims>
ddc::DiscreteDomain<DDims...> optional_initialize_dims_periodic_unit_cube(
    ddc::DiscreteVector<DDims...> &resolution) {
  return ddc::DiscreteDomain<DDims...>(
      optional_initialize_dims_periodic_unit_intervals<DDims>(resolution)...);
}

} // namespace paliwa