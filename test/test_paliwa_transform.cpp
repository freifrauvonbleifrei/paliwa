
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
  for (long int min_level_x : {0, 1, 3}) {
    for (long int min_level_y : {0, 1, 3}) {
      DVect const minimum_level(
          std::array<long int, 2>({min_level_x, min_level_y}));
      SDDom strided_domain = // indices 0, 2, ...14
          paliwa::strided_domain_from_level<DDimX, DDimY>(
              ddc::detail::array(level), ddc::detail::array(maximum_level));
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

      paliwa::hierarchize(strided_grid, level, minimum_level, maximum_level,
                          "hat", Kokkos::DefaultExecutionSpace());
      auto strided_grid_host = ddc::create_mirror_view_and_copy(
          Kokkos::SharedHostPinnedSpace(), strided_grid);
      SDDom const strided_domain_lmin =
          paliwa::strided_domain_from_level<DDimX, DDimY>(
              ddc::detail::array(minimum_level),
              ddc::detail::array(maximum_level));
      ddc::host_for_each(strided_domain, [&](DElem const ixy) {
        if (strided_domain_lmin.contains(ixy)) {
          EXPECT_NEAR(strided_grid_host(ixy), strided_grid_host_before(ixy),
                      1e-14);
        } else {
          EXPECT_NE(strided_grid_host(ixy), strided_grid_host_before(ixy));
        }
      });

      paliwa::dehierarchize(strided_grid, level, minimum_level, maximum_level,
                            "hat", Kokkos::DefaultExecutionSpace());
      auto strided_grid_host_after = ddc::create_mirror_view_and_copy(
          Kokkos::SharedHostPinnedSpace(), strided_grid);
      ddc::host_for_each(strided_domain, [&](DElem const ixy) {
        EXPECT_NEAR(strided_grid_host_after(ixy), strided_grid_host_before(ixy),
                    1e-13);
      });
    }
  }
}
// googletest / KOKKOS_LAMBDA workaround
// https://github.com/kokkos/kokkos-comm/pull/72
TEST(transform, test_hat_stays_same_2d) { test_transform_hat_stays_same_2d(); }

void test_transform_mass_conservation_2d(std::string wavelet_name) {
  using DDom = ddc::DiscreteDomain<DDimX, DDimY>;
  using SDDom = ddc::StridedDiscreteDomain<DDimX, DDimY>;
  using DElem = DDom::discrete_element_type;
  using DVect = SDDom::discrete_vector_type;
  DVect const level(std::array<long int, 2>({3, 3}));
  DVect const maximum_level(std::array<long int, 2>({4, 4}));
  for (long int min_level_x : {0, 1, 3}) {
    for (long int min_level_y : {0, 1, 3}) {
      DVect const minimum_level(
          std::array<long int, 2>({min_level_x, min_level_y}));
      SDDom strided_domain = // indices 0, 2, ...14
          paliwa::strided_domain_from_level<DDimX, DDimY>(
              ddc::detail::array(level), ddc::detail::array(maximum_level));
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

      double const reference_average =
          ddc::parallel_transform_reduce(
              Kokkos::DefaultExecutionSpace(), strided_domain, 0.0,
              ddc::reducer::sum<double>(), strided_grid) /
          strided_domain.size();
      paliwa::hierarchize(strided_grid, level, minimum_level, maximum_level,
                          wavelet_name, Kokkos::DefaultExecutionSpace());
      // // for lmin [0, 0], can check the first entry too
      // auto strided_grid_host = ddc::create_mirror_view_and_copy(
      //     Kokkos::SharedHostPinnedSpace(), strided_grid);
      // double coeff_00 = strided_grid_host(strided_domain.front());
      // EXPECT_NEAR(coeff_00, reference_average, 1e-10);

      SDDom const strided_domain_lmin =
          paliwa::strided_domain_from_level<DDimX, DDimY>(
              ddc::detail::array(minimum_level),
              ddc::detail::array(maximum_level));
      double lmin_sum = ddc::parallel_transform_reduce(
          Kokkos::DefaultExecutionSpace(), strided_domain_lmin, 0.0,
          ddc::reducer::sum<double>(), strided_grid);
      EXPECT_NEAR(lmin_sum, reference_average * strided_domain_lmin.size(),
                  1e-13);
    }
  }
}
TEST(transform, test_mass_conservation_2d) {
  for (std::string wavelet_name :
       paliwa::lifting_wavelet_filter_offsets_and_coefficients |
           std::views::keys) {
    if (wavelet_name == "hat")
      continue;
    test_transform_mass_conservation_2d(wavelet_name);
  }
}