// SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-2.1-or-later

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "../src/paliwa_domains.hpp"

struct X {};
struct Y {};
struct DDimX : ddc::UniformPointSampling<X> {};
struct DDimY : ddc::UniformPointSampling<Y> {};

TEST(domain, restrict_strided_domain_from_domain) {
  using DDom = ddc::DiscreteDomain<DDimX, DDimY>;
  using SDDom = ddc::StridedDiscreteDomain<DDimX, DDimY>;
  using DElem = DDom::discrete_element_type;
  SDDom strided_dom_all = // indices 0, 2, ...14
      paliwa::strided_domain_from_level<DDimX, DDimY>({3, 3}, {4, 4});
  DDom local_dom =
      DDom(ddc::DiscreteDomain<DDimX>(ddc::DiscreteElement<DDimX>(1),
                                      ddc::DiscreteVector<DDimX>(5)),
           // indices 1 through 5
           ddc::DiscreteDomain<DDimY>(ddc::DiscreteElement<DDimY>(4),
                                      ddc::DiscreteVector<DDimY>(9)));
  // indices 4 through 12
  SDDom restricted_strided_dom =
      paliwa::restrict_strided_with_discrete(strided_dom_all, local_dom);
  EXPECT_EQ(restricted_strided_dom.front(), DElem(std::array<int, 2>({2, 4})));
  EXPECT_EQ(restricted_strided_dom.back(),
            DElem(std::array<int, 2>({4, 12}))); // every other in x-direction

  // indices 4 through 11
  ddc::DiscreteDomain<DDimY> const y_local_dom = ddc::DiscreteDomain<DDimY>(
      ddc::DiscreteElement<DDimY>(4), ddc::DiscreteVector<DDimY>(8));
  SDDom restricted_strided_dom_2 =
      paliwa::restrict_strided_with_discrete(strided_dom_all, y_local_dom);
  EXPECT_EQ(restricted_strided_dom_2.front(),
            DElem(std::array<int, 2>({0, 4})));
  EXPECT_EQ(restricted_strided_dom_2.back(),
            DElem(std::array<int, 2>({14, 10})));
}