
#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "../src/paliwa_distribute.hpp"
#include "../src/paliwa_domains.hpp"

struct X {};
struct Y {};
struct DDimX : ddc::UniformPointSampling<X> {};
struct DDimY : ddc::UniformPointSampling<Y> {};

void test_transform_hat_stays_same_2d() {
  using DDom = ddc::DiscreteDomain<DDimX, DDimY>;
  using SDDom = ddc::StridedDiscreteDomain<DDimX, DDimY>;
  using DElem = DDom::discrete_element_type;
  using DVect = SDDom::discrete_vector_type;
  DVect const level(std::array<long int, 2>({3, 3}));

  DVect const maximum_level(std::array<long int, 2>({4, 4}));
  for (long int min_level_x : {0, 1, 2, 3}) { // todo
    DVect const minimum_level(std::array<long int, 2>({min_level_x, 0}));
    SDDom strided_domain = // indices 0, 2, ...14
        paliwa::strided_domain_from_level(ddc::detail::array(level),
                                          ddc::detail::array(maximum_level),
                                          DElem({}));
    ddc::Chunk strided_grid_chunk(strided_domain,
                                  ddc::DeviceAllocator<double>());
    auto strided_grid = strided_grid_chunk.span_view();
    // fill with random data
    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
    ddc::parallel_for_each(
        Kokkos::DefaultExecutionSpace(), strided_domain,
        KOKKOS_LAMBDA(DElem const ixy) {
          auto generator = random_pool.get_state();
          double random_number = generator.drand(0., 1.);
          random_pool.free_state(generator);
          strided_grid(ixy) =
              (1 + ixy.uid<DDimX>() + ixy.uid<DDimY>()) * (1 + random_number);
        });
    auto strided_grid_host_before = ddc::create_mirror_view_and_copy(
        Kokkos::SharedHostPinnedSpace(), strided_grid);

    paliwa::hierarchize(strided_domain, strided_grid, level, minimum_level,
                        maximum_level, DElem({}), "hat",
                        Kokkos::DefaultExecutionSpace());
    auto strided_grid_host = ddc::create_mirror_view_and_copy(
        Kokkos::SharedHostPinnedSpace(), strided_grid);
    SDDom const strided_domain_lmin = paliwa::strided_domain_from_level(
        ddc::detail::array(minimum_level), ddc::detail::array(maximum_level),
        DElem({}));
    ddc::host_for_each(strided_domain, [&](DElem const ixy) {
      if (strided_domain_lmin.contains(ixy)) {
        EXPECT_EQ(strided_grid_host(ixy), strided_grid_host_before(ixy));
      } else {
        EXPECT_NOT_EQ(strided_grid_host(ixy), strided_grid_host_before(ixy));
      }
    });

    paliwa::dehierarchize(strided_domain, strided_grid, level, minimum_level,
                          maximum_level, DElem({}), "hat",
                          Kokkos::DefaultExecutionSpace());
    auto strided_grid_host_after = ddc::create_mirror_view_and_copy(
        Kokkos::SharedHostPinnedSpace(), strided_grid);
    ddc::host_for_each(strided_domain, [&](DElem const ixy) {
      EXPECT_NEAR(strided_grid_host_after(ixy), strided_grid_host_before(ixy),
                  1e-13);
    });
  }
}
// googletest / KOKKOS_LAMBDA workaround
// https://github.com/kokkos/kokkos-comm/pull/72
TEST(transform, test_hat_stays_same_2d) { test_transform_hat_stays_same_2d(); }
