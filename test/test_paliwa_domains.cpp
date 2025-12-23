// SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-2.1-or-later

#include <Kokkos_Random.hpp>
#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "../src/paliwa_domains.hpp"

struct X {};
struct DDimX : ddc::UniformPointSampling<X> {};

TEST(domain, sparse_domain_conversion_1d) {
  using DElem = ddc::DiscreteElement<DDimX>;
  ddc::DiscreteDomain<DDimX> local_domain =
      ddc::DiscreteDomain<DDimX>(DElem(1), ddc::DiscreteVector<DDimX>(200));

  Kokkos::View<DElem *, Kokkos::SharedSpace> elements("elements",
                                                      local_domain.size());

  Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace> random_pool(
      /*seed=*/12345);
  size_t insert_index = 0;
  ddc::host_for_each(
      local_domain, [&elements, &random_pool, &insert_index](DElem ixyz) {
        auto generator = random_pool.get_state();
        double random_number = generator.drand(0., 1.);
        random_pool.free_state(generator);
        if (random_number < 0.4) {
          elements(insert_index++) = ixyz;
        }
      });
  Kokkos::resize(elements, insert_index);
  ddc::SparseDiscreteDomain<DDimX> sparse_domain(elements);

  ddc::StridedDiscreteDomain<DDimX> strided_domain(
      DElem(20), ddc::DiscreteVector<DDimX>(20), ddc::DiscreteVector<DDimX>(3));

  ddc::SparseDiscreteDomain<DDimX> sparse_and_strided_domain =
      paliwa::union_of_sparse_domains(
          paliwa::sparse_from_strided_domain(strided_domain), sparse_domain);

  ddc::host_for_each(strided_domain, [&sparse_and_strided_domain](DElem ixyz) {
    EXPECT_TRUE(sparse_and_strided_domain.contains(ixyz));
  });
  ddc::host_for_each(sparse_domain, [&sparse_and_strided_domain](DElem ixyz) {
    EXPECT_TRUE(sparse_and_strided_domain.contains(ixyz));
  });
}