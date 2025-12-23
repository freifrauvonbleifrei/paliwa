// SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <Kokkos_Random.hpp>
#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "../src/paliwa_domains.hpp"

struct X {};
struct Y {};
struct DDimX : ddc::UniformPointSampling<X> {};
struct DDimY : ddc::UniformPointSampling<Y> {};

template <typename DDim>
ddc::SparseDiscreteDomain<DDim>
random_sparse_subdomain(ddc::DiscreteDomain<DDim> local_domain) {
  using DElem = ddc::DiscreteElement<DDim>;
  Kokkos::View<DElem *, Kokkos::SharedSpace> elements("elements",
                                                      local_domain.size());

  Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace> random_pool(
      /*seed=*/12345);
  size_t insert_index = 0;
  ddc::host_for_each(local_domain,
                     [&elements, &random_pool, &insert_index](DElem ixyz) {
                       auto generator = random_pool.get_state();
                       double random_number = generator.drand(0., 1.);
                       random_pool.free_state(generator);
                       if (random_number < 0.4) {
                         elements(insert_index++) = ixyz;
                       }
                     });
  Kokkos::resize(elements, insert_index);
  return ddc::SparseDiscreteDomain<DDim>(elements);
}

TEST(domain, sparse_domain_conversion_1d) {
  using DElem = ddc::DiscreteElement<DDimX>;
  ddc::DiscreteDomain<DDimX> local_domain =
      ddc::DiscreteDomain<DDimX>(DElem(1), ddc::DiscreteVector<DDimX>(200));

  ddc::SparseDiscreteDomain<DDimX> sparse_domain =
      random_sparse_subdomain(local_domain);

  ddc::StridedDiscreteDomain<DDimX> strided_domain(
      DElem(20), ddc::DiscreteVector<DDimX>(20), ddc::DiscreteVector<DDimX>(3));

  // test union
  ddc::SparseDiscreteDomain<DDimX> sparse_and_strided_domain =
      paliwa::union_of_sparse_domains(
          paliwa::sparse_from_strided_domain(strided_domain), sparse_domain);

  ddc::host_for_each(strided_domain, [&sparse_and_strided_domain](DElem ixyz) {
    EXPECT_TRUE(sparse_and_strided_domain.contains(ixyz));
  });
  ddc::host_for_each(sparse_domain, [&sparse_and_strided_domain](DElem ixyz) {
    EXPECT_TRUE(sparse_and_strided_domain.contains(ixyz));
  });

  // also test intersection
  ddc::SparseDiscreteDomain<DDimX> restricted_sparse_domain =
      paliwa::restrict_sparse_with_other_domain(sparse_domain, strided_domain);

  ddc::host_for_each(sparse_domain,
                     [&strided_domain, &restricted_sparse_domain](DElem ixyz) {
                       if (strided_domain.contains(ixyz)) {
                         EXPECT_TRUE(restricted_sparse_domain.contains(ixyz));
                       } else {
                         EXPECT_FALSE(restricted_sparse_domain.contains(ixyz));
                       }
                     });
}

TEST(domain, sparse_domain_conversion_2d) {
  using DElem = ddc::DiscreteElement<DDimX, DDimY>;
  ddc::DiscreteDomain<DDimX, DDimY> local_domain(
      ddc::DiscreteDomain<DDimX>(ddc::DiscreteElement<DDimX>(1),
                                 ddc::DiscreteVector<DDimX>(20)),
      ddc::DiscreteDomain<DDimY>(ddc::DiscreteElement<DDimY>(5),
                                 ddc::DiscreteVector<DDimY>(30)));
  ddc::SparseDiscreteDomain<DDimX, DDimY> sparse_domain(
      random_sparse_subdomain(ddc::select<DDimX>(local_domain)),
      random_sparse_subdomain(ddc::select<DDimY>(local_domain)));

  ddc::StridedDiscreteDomain<DDimX, DDimY> strided_domain(
      ddc::DiscreteElement<DDimX, DDimY>(std::array<int, 2>({10, 10})),
      ddc::DiscreteVector<DDimX, DDimY>(std::array<long int, 2>({10, 10})),
      ddc::DiscreteVector<DDimX, DDimY>(std::array<long int, 2>({2, 3})));
  // test union
  ddc::SparseDiscreteDomain<DDimX, DDimY> sparse_and_strided_domain =
      paliwa::union_of_sparse_domains(
          paliwa::sparse_from_strided_domain(strided_domain), sparse_domain);
  ddc::host_for_each(strided_domain, [&sparse_and_strided_domain](DElem ixyz) {
    EXPECT_TRUE(sparse_and_strided_domain.contains(ixyz));
  });
  ddc::host_for_each(sparse_domain, [&sparse_and_strided_domain](DElem ixyz) {
    EXPECT_TRUE(sparse_and_strided_domain.contains(ixyz));
  });
//   auto first_dim_sparse_and_strided = //TODO why does this fail? -> needed to make the rest work
//       ddc::select<DDimX>(sparse_and_strided_domain);
  //   // also test intersection
  //   ddc::SparseDiscreteDomain<DDimX, DDimY> restricted_sparse_domain =
  //       paliwa::restrict_sparse_with_other_domain(sparse_domain,
  //       strided_domain);
  //   ddc::host_for_each(sparse_domain,
  //                      [&strided_domain, &restricted_sparse_domain](DElem
  //                      ixyz) {
  //                        if (strided_domain.contains(ixyz)) {
  //                          EXPECT_TRUE(restricted_sparse_domain.contains(ixyz));
  //                        } else {
  //                          EXPECT_FALSE(restricted_sparse_domain.contains(ixyz));
  //                        }
  //                      });
}