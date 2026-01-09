// SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <cstdlib> // srand etc

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "../paliwa/paliwa_domains.hpp"
#include "../paliwa/paliwa_transform.hpp"

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
void get_required_transform_domains_1d() {
  using DDom = ddc::DiscreteDomain<DDimX>;
  using SDDom = ddc::StridedDiscreteDomain<DDimX>;
  using DElem = DDom::discrete_element_type;
  using DVect = SDDom::discrete_vector_type;
  for (std::string wavelet_name :
       paliwa::lifting_wavelet_filter_offsets_and_coefficients |
           std::views::keys) {
    SCOPED_TRACE(wavelet_name);
    for (bool is_for_hierarchization : {true, false}) {
      SCOPED_TRACE(is_for_hierarchization ? "hierarchization"
                                          : "dehierarchization");
      DVect const level(std::array<long int, 1>({6}));
      DVect const maximum_level(std::array<long int, 1>({7}));
      for (long int lmin = 0; lmin < level.template get<DDimX>(); ++lmin) {
        SCOPED_TRACE("lmin=" + std::to_string(lmin));
        DVect const minimum_level(std::array<long int, 1>({lmin}));
        SDDom strided_dom_all = // indices 0, 2, ...126
            paliwa::strided_domain_from_level<DDimX>(
                ddc::detail::array(level), ddc::detail::array(maximum_level));
        // randomly select the local domain's bounds
        srand(time(0));
        int random_start = rand() % 126;
        int random_extent = 1 + rand() % (125 - random_start);
        DDom local_dom = DDom(ddc::DiscreteDomain<DDimX>(
            ddc::DiscreteElement<DDimX>(random_start),
            ddc::DiscreteVector<DDimX>(random_extent)));
        SDDom restricted_strided_dom =
            paliwa::restrict_strided_with_discrete(strided_dom_all, local_dom);
        ddc::SparseDiscreteDomain<DDimX> required_transform_domain =
            paliwa::get_required_transform_domains(
                is_for_hierarchization, strided_dom_all, restricted_strided_dom,
                level, minimum_level, maximum_level, wavelet_name);

        // combine sparse and restricted strided into common domain
        ddc::SparseDiscreteDomain<DDimX> required_transform_and_local_domain =
            paliwa::union_of_sparse_domains(
                paliwa::sparse_from_strided_domain(restricted_strided_dom),
                required_transform_domain);

        ddc::Chunk all_strided_chunk(strided_dom_all,
                                     ddc::HostAllocator<float>());
        auto all_strided_span = all_strided_chunk.span_view();
        ddc::Chunk transform_chunk(required_transform_and_local_domain,
                                   ddc::HostAllocator<float>());
        auto transform_span = transform_chunk.span_view();
        // fill with NaNs
        ddc::host_for_each(
            strided_dom_all, KOKKOS_LAMBDA(DElem ixyz) {
              all_strided_span(ixyz) = std::nanf("");
            });
        // fill local with 1.0s
        ddc::host_for_each(
            restricted_strided_dom, KOKKOS_LAMBDA(DElem ixyz) {
              all_strided_span(ixyz) = 1.0;
              transform_span(ixyz) = 1.0;
            });
        // fill required transform domain with 2.0s
        ddc::host_for_each(
            required_transform_domain, KOKKOS_LAMBDA(DElem ixyz) {
              all_strided_span(ixyz) = 2.0;
              transform_span(ixyz) = 2.0;
            });
        // do the transform and check non-nans
        if (is_for_hierarchization) {
          paliwa::hierarchize(all_strided_span, level, minimum_level,
                              maximum_level, wavelet_name,
                              Kokkos::DefaultHostExecutionSpace());
        } else {
          paliwa::dehierarchize(all_strided_span, level, minimum_level,
                                maximum_level, wavelet_name,
                                Kokkos::DefaultHostExecutionSpace());
        }
        ddc::host_for_each(
            restricted_strided_dom, KOKKOS_LAMBDA(DElem ixyz) {
              ASSERT_FALSE(std::isnan(all_strided_span(ixyz)));
            });

        // do the transform again and check for same result
        if (is_for_hierarchization) {
          paliwa::hierarchize(transform_span, level, minimum_level,
                              maximum_level, wavelet_name,
                              Kokkos::DefaultHostExecutionSpace());
        } else {
          paliwa::dehierarchize(transform_span, level, minimum_level,
                                maximum_level, wavelet_name,
                                Kokkos::DefaultHostExecutionSpace());
        }
        ddc::host_for_each(
            restricted_strided_dom, KOKKOS_LAMBDA(DElem ixyz) {
              EXPECT_EQ(all_strided_span(ixyz), transform_span(ixyz));
            });
      }
    }
  }
}
// googletest / KOKKOS_LAMBDA workaround
// https://github.com/kokkos/kokkos-comm/pull/72
TEST(domain, get_required_transform_domains_1d) {
  get_required_transform_domains_1d();
}