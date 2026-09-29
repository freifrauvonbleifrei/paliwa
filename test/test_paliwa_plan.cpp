// SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <complex>
#include <limits>
#include <random>
#include <set>

#include <gtest/gtest.h>

#include "../paliwa/paliwa_transform.hpp"

namespace {
struct PlanAxis {};
struct PlanDim : ddc::UniformPointSampling<PlanAxis> {};
using Filters = std::vector<std::pair<int, std::array<double, 3>>>;

std::set<long> points(paliwa::PeriodicBounds b, long stride, long period) {
  std::set<long> result;
  if (!b.empty()) {
    long last = b.max + (b.min > b.max ? period : 0);
    for (long i = b.min; i <= last; i += stride)
      result.insert(i % period);
  }
  return result;
}

void check_plan(long level, long maximum, long start, long count,
                std::vector<long> const &levels, Filters const &filters) {
  long const period = 1L << maximum;
  long const spacing = 1L << (maximum - level);
  paliwa::PeriodicBounds output;
  if (count)
    output = {start * spacing, ((start + count - 1) * spacing) % period};
  auto required = points(output, spacing, period);
  std::vector<std::vector<std::set<long>>> expected(
      levels.size(), std::vector<std::set<long>>(filters.size()));
  // Independent reference: reverse the actual stencil graph, without using
  // interval operations or inspecting the numerical filter coefficients.
  for (size_t l = levels.size(); l-- > 0;) {
    long const stride = 1L << (maximum - levels[l]);
    for (size_t p = filters.size(); p-- > 0;) {
      for (long i : required)
        if (i % (2 * stride) == filters[p].first * stride)
          expected[l][p].insert(i);
      for (long i : expected[l][p]) {
        required.insert((i + period - stride) % period);
        required.insert((i + stride) % period);
      }
    }
  }
  auto const bounds = paliwa::transform_bounds(
      output, level, maximum, levels, filters,
      [&](size_t l, size_t p, paliwa::PeriodicBounds writes) {
        EXPECT_EQ(points(writes, 2L << (maximum - levels[l]), period),
                  expected[l][p]);
      });
  ASSERT_EQ(bounds.size(), static_cast<size_t>(level + 1));
  std::set<long> actual;
  for (long l = 0; l <= level; ++l) {
    auto const values =
        points(bounds[l], l == 0 ? period : 2L << (maximum - l), period);
    actual.insert(values.begin(), values.end());
  }
  EXPECT_EQ(actual, required);
}
} // namespace

TEST(transform_plan, exact_access_sets) {
  // Exhaust every parity sequence up to five passes, every wrapped/ordinary
  // output interval, both directions, partial levels, and a coarser input grid.
  for (long maximum : {4, 6}) {
    for (size_t passes = 1; passes <= 5; ++passes) {
      for (unsigned mask = 0; mask < (1U << passes); ++mask) {
        Filters filters;
        for (size_t p = 0; p < passes; ++p)
          filters.push_back({(mask & (1U << p)) ? 1 : 0, {0., 1., 0.}});
        for (auto const &levels : std::vector<std::vector<long>>{
                 {4, 3, 2, 1}, {1, 2, 3, 4}, {4, 3}, {3, 4}, {4}, {}}) {
          for (long start = 0; start < 16; ++start) {
            for (long count = 0; count <= 16; ++count) {
              SCOPED_TRACE(::testing::Message()
                           << maximum << ':' << passes << ':' << mask << ':'
                           << start << ':' << count);
              check_plan(4, maximum, start, count, levels, filters);
            }
          }
        }
      }
    }
  }
}

TEST(transform_plan, longer_filter_sequences) {
  std::mt19937 rng(20260928);
  for (int trial = 0; trial < 2000; ++trial) {
    long const level = 1 + rng() % 8;
    long const minimum = rng() % (level + 1);
    std::vector<long> levels;
    for (long l = minimum + 1; l <= level; ++l)
      levels.push_back(l);
    if (rng() % 2)
      std::reverse(levels.begin(), levels.end());
    Filters filters;
    for (unsigned p = 0, n = 6 + rng() % 7; p < n; ++p)
      filters.push_back({static_cast<int>(rng() % 2), {0., 0., 0.}});
    SCOPED_TRACE(trial);
    check_plan(level, level + 2, rng() % (1L << level),
               rng() % ((1L << level) + 1), levels, filters);
  }
}

TEST(transform_plan, multipass_sparse_values) {
  using Element = ddc::DiscreteElement<PlanDim>;
  using Vector = ddc::DiscreteVector<PlanDim>;
  Kokkos::DefaultHostExecutionSpace exec;
  constexpr long maximum = 6, level = 4, spacing = 4, period = 64;
  for (size_t passes = 1; passes <= 5; ++passes) {
    for (unsigned mask = 0; mask < (1U << passes); ++mask) {
      Filters filters;
      for (size_t p = 0; p < passes; ++p)
        filters.push_back({(mask & (1U << p)) ? 1 : 0,
                           p % 2 ? std::array<double, 3>{0., 1., .25}
                                 : std::array<double, 3>{-.5, 1., .125}});
      for (auto const &levels :
           std::vector<std::vector<long>>{{4, 3, 2, 1}, {1, 2, 3, 4}}) {
        std::array<double, period> input{}, reference{};
        for (long i = 0; i < period; ++i)
          input[i] = 1. + std::sin(i * .7);
        reference = input;
        for (long l : levels) {
          long const stride = 1L << (maximum - l);
          for (auto const &[offset, f] : filters)
            for (long i = offset * stride; i < period; i += 2 * stride)
              reference[i] = f[0] * reference[(i + period - stride) % period] +
                             f[1] * reference[i] +
                             f[2] * reference[(i + stride) % period];
        }
        for (long start : {0, 3, 12, 15}) {
          long const count = std::min(4L, 16 - start);
          ddc::StridedDiscreteDomain<PlanDim> outputs(
              Element(start * spacing), Vector(count), Vector(spacing));
          auto bounds = paliwa::transform_bounds(
              {start * spacing, (start + count - 1) * spacing}, level, maximum,
              levels, filters, [](size_t, size_t, paliwa::PeriodicBounds) {});
          auto domain =
              paliwa::domain_from_transform_bounds<PlanDim>(bounds, maximum);
          ddc::Chunk chunk(domain, ddc::HostAllocator<double>());
          auto grid = chunk.span_view();
          for (auto order : {paliwa::PoleExecution::Direct,
                             paliwa::PoleExecution::PackedPasses,
                             paliwa::PoleExecution::PackedPoles}) {
            ddc::host_for_each(
                domain, [&](Element i) { grid(i) = input[i.uid<PlanDim>()]; });
            paliwa::transform_in<PlanDim>(grid, Vector(level), Vector(maximum),
                                          levels, filters, outputs, exec,
                                          order);
            exec.fence();
            ddc::host_for_each(outputs, [&](Element i) {
              EXPECT_NEAR(grid(i), reference[i.uid<PlanDim>()], 1e-12);
            });
          }
          // The same exact plan with local strided data and remote-only
          // sparse storage. Ghost values must remain unchanged.
          Kokkos::View<Element *, Kokkos::SharedSpace> ghosts("ghosts",
                                                              domain.size());
          size_t n = 0;
          ddc::host_for_each(domain, [&](Element i) {
            if (!outputs.contains(i))
              ghosts(n++) = i;
          });
          Kokkos::resize(ghosts, n);
          ddc::SparseDiscreteDomain<PlanDim> remote_domain(ghosts);
          ddc::Chunk local_chunk(outputs, ddc::HostAllocator<double>());
          ddc::Chunk remote_chunk(remote_domain, ddc::HostAllocator<double>());
          auto local = local_chunk.span_view();
          auto remote = remote_chunk.span_view();
          ddc::host_for_each(
              outputs, [&](Element i) { local(i) = input[i.uid<PlanDim>()]; });
          ddc::host_for_each(remote_domain, [&](Element i) {
            remote(i) = input[i.uid<PlanDim>()];
          });
          paliwa::detail::transform_with_remote<PlanDim>(
              local, remote, Vector(level), Vector(maximum), levels, filters,
              outputs, exec, paliwa::PoleExecution::PackedPoles);
          ddc::host_for_each(outputs, [&](Element i) {
            EXPECT_NEAR(local(i), reference[i.uid<PlanDim>()], 1e-12);
          });
          ddc::host_for_each(remote_domain, [&](Element i) {
            EXPECT_EQ(remote(i), input[i.uid<PlanDim>()]);
          });
        }
      }
    }
  }
}

TEST(transform_plan, nonconsecutive_levels_are_rejected) {
  Filters const filters{{1, {0., 1., 0.}}};
  auto ignore = [](size_t, size_t, paliwa::PeriodicBounds) {};
  for (auto const &levels :
       std::vector<std::vector<long>>{{4, 1}, {1, 4}, {1, 2, 1}}) {
    EXPECT_THROW(
        paliwa::transform_bounds({0, 0}, 4, 4, levels, filters, ignore),
        std::invalid_argument);
  }
}

TEST(transform_plan, periodic_pole_cover) {
  auto check = [](std::vector<long> indices, long min, long max) {
    auto b = paliwa::covering_periodic_bounds(indices, 64);
    EXPECT_EQ(b.min, min);
    EXPECT_EQ(b.max, max);
  };
  check({}, -1, -1);
  check({12}, 12, 12);
  check({8, 12, 16}, 8, 16);
  check({0, 4, 56, 60}, 56, 4);
  check({0, 16, 32, 48}, 0, 48);
}

namespace {
struct PoleAxis {};
struct PoleDim : ddc::UniformPointSampling<PoleAxis> {};
} // namespace

TEST(transform_plan, packed_complex_poles) {
  using Value = std::complex<double>;
  using Vector = ddc::DiscreteVector<PlanDim, PoleDim>;
  using Element = ddc::DiscreteElement<PlanDim, PoleDim>;
  Vector const level(4, 3), maximum(6, 5);
  auto domain = paliwa::strided_domain_from_level<PlanDim, PoleDim>(
      ddc::detail::array(level), ddc::detail::array(maximum));
  ddc::Chunk reference(domain, ddc::HostAllocator<Value>());
  ddc::Chunk candidate(domain, ddc::HostAllocator<Value>());
  auto expected = reference.span_view(), actual = candidate.span_view();
  auto reset = [&](auto grid) {
    ddc::host_for_each(domain, [&](Element i) {
      grid(i) = Value(std::sin(i.uid<PlanDim>() + .3 * i.uid<PoleDim>()),
                      std::cos(.7 * i.uid<PlanDim>() - i.uid<PoleDim>()));
    });
  };
  Filters const filters{{1, {-.5, 1., .125}},
                        {0, {0., 1., .25}},
                        {0, {.125, 1., .125}},
                        {1, {0., 1., .25}},
                        {1, {-.25, 1., .125}}};
  Kokkos::DefaultHostExecutionSpace exec;
  auto check = [&]<typename Dim>() {
    long const n = static_cast<long>(ddc::select<Dim>(level));
    std::vector<long> levels;
    for (long l = n; l > 0; --l)
      levels.push_back(l);
    for (bool inverse : {false, true}) {
      if (inverse)
        std::reverse(levels.begin(), levels.end());
      reset(expected);
      paliwa::transform_in<Dim>(expected, level, maximum, levels, filters,
                                domain, exec, paliwa::PoleExecution::Direct);
      for (auto order : {paliwa::PoleExecution::PackedPasses,
                         paliwa::PoleExecution::PackedPoles}) {
        reset(actual);
        paliwa::transform_in<Dim>(actual, level, maximum, levels, filters,
                                  domain, exec, order);
        exec.fence();
        ddc::host_for_each(domain, [&](Element i) {
          EXPECT_NEAR(std::abs(actual(i) - expected(i)), 0., 1e-12);
        });
      }
    }
  };
  check.template operator()<PlanDim>();
  check.template operator()<PoleDim>();
}

TEST(transform_plan, remote_only_complex_poles) {
  using Value = std::complex<double>;
  using Element = ddc::DiscreteElement<PlanDim, PoleDim>;
  using Vector = ddc::DiscreteVector<PlanDim, PoleDim>;
  Vector const level(4, 3), maximum(6, 5);
  auto const full = paliwa::strided_domain_from_level<PlanDim, PoleDim>(
      ddc::detail::array(level), ddc::detail::array(maximum));
  Filters const filters{{1, {-.5, 1., .125}},
                        {0, {0., 1., .25}},
                        {0, {.125, 1., .125}},
                        {1, {0., 1., .25}},
                        {1, {-.25, 1., .125}}};
  auto value = [](Element e) {
    return Value(std::sin(e.uid<PlanDim>() + .3 * e.uid<PoleDim>()),
                 std::cos(.7 * e.uid<PlanDim>() - e.uid<PoleDim>()));
  };
  Kokkos::DefaultHostExecutionSpace exec;
  auto check = [&]<typename Dim>() {
    long const n = static_cast<long>(ddc::select<Dim>(level));
    auto const axis = ddc::select<Dim>(full);
    std::vector<long> levels;
    for (long l = n; l > 0; --l)
      levels.push_back(l);
    for (bool inverse : {false, true}) {
      if (inverse)
        std::reverse(levels.begin(), levels.end());
      ddc::Chunk reference_chunk(full, ddc::HostAllocator<Value>());
      auto reference = reference_chunk.span_view();
      ddc::host_for_each(full, [&](Element e) { reference(e) = value(e); });
      paliwa::transform_in<Dim>(reference, level, maximum, levels, filters,
                                full, exec, paliwa::PoleExecution::Direct);
      // First and last slabs exercise periodic dependencies. The full slab
      // exercises an empty remote allocation; both dimension orders are used.
      long const size = static_cast<long>(axis.size());
      for (auto [start, count] : std::vector<std::pair<long, long>>{
               {0, size / 2}, {size / 2, size / 2}, {0, size}}) {
        ddc::StridedDiscreteDomain<Dim> local_axis(
            axis.front() + axis.strides() * start,
            ddc::DiscreteVector<Dim>(count), axis.strides());
        ddc::StridedDiscreteDomain<PlanDim, PoleDim> outputs(
            local_axis, ddc::remove_dims_of<Dim>(full));
        auto sparse_axis = [&](auto tag) {
          using Axis = decltype(tag);
          auto const source = ddc::select<Axis>(full);
          Kokkos::View<ddc::DiscreteElement<Axis> *, Kokkos::SharedSpace>
              indices("indices", source.size());
          size_t count = 0;
          ddc::host_for_each(source, [&](auto e) {
            if constexpr (std::is_same_v<Axis, Dim>) {
              if (local_axis.contains(e))
                return;
            }
            indices(count++) = e;
          });
          Kokkos::resize(indices, count);
          return ddc::SparseDiscreteDomain<Axis>(indices);
        };
        ddc::SparseDiscreteDomain<PlanDim, PoleDim> remote_domain(
            sparse_axis(PlanDim{}), sparse_axis(PoleDim{}));
        ddc::Chunk local_chunk(outputs, ddc::HostAllocator<Value>());
        ddc::Chunk remote_chunk(remote_domain, ddc::HostAllocator<Value>());
        auto local = local_chunk.span_view();
        auto remote = remote_chunk.span_view();
        ddc::host_for_each(outputs, [&](Element e) { local(e) = value(e); });
        ddc::host_for_each(remote_domain,
                           [&](Element e) { remote(e) = value(e); });
        paliwa::detail::transform_with_remote<Dim>(
            local, remote, level, maximum, levels, filters, outputs, exec,
            paliwa::PoleExecution::PackedPoles);
        ddc::host_for_each(outputs, [&](Element e) {
          EXPECT_NEAR(std::abs(local(e) - reference(e)), 0., 1e-12);
        });
        ddc::host_for_each(remote_domain,
                           [&](Element e) { EXPECT_EQ(remote(e), value(e)); });
      }
    }
  };
  check.template operator()<PlanDim>();
  check.template operator()<PoleDim>();
}

TEST(transform_plan, packed_pass_exact_accesses) {
  EXPECT_TRUE(paliwa::pole_pass_segments({}, 0, 3, 5, 2).empty());
  // Compare ordered (destination, lower, upper) slot triples against global
  // periodic indexing for every origin, parity, and interval on small grids.
  for (long level = 1; level <= 6; ++level) {
    long const period = 1L << level;
    long const spacing = 4;
    for (long pass = 1; pass <= level; ++pass) {
      long const distance = 1L << (level - pass);
      for (long origin = 0; origin < period; ++origin) {
        for (long parity : {0L, 1L}) {
          for (long start = parity * distance; start < period;
               start += 2 * distance) {
            for (long count = 1; count <= period / (2 * distance); ++count) {
              long const last = (start + (count - 1) * 2 * distance) % period;
              auto parts = paliwa::pole_pass_segments(
                  {start * spacing, last * spacing}, origin * spacing, level,
                  level + 2, pass);
              std::vector<std::array<int, 3>> actual, expected;
              for (auto const &part : parts) {
                for (decltype(part.count) i = 0; i < part.count; ++i) {
                  auto const offset = i * part.stride;
                  actual.push_back({part.first + offset, part.lower + offset,
                                    part.upper + offset});
                }
              }
              auto slot = [&](long x) {
                return static_cast<int>((x - origin + 2 * period) % period);
              };
              for (long i = 0; i < count; ++i) {
                long const x = (start + i * 2 * distance) % period;
                expected.push_back(
                    {slot(x), slot(x - distance), slot(x + distance)});
              }
              ASSERT_EQ(actual, expected)
                  << level << ':' << pass << ':' << origin << ':' << start
                  << ':' << count;
            }
          }
        }
      }
    }
  }
}

TEST(transform_plan, packed_pass_large_coordinates) {
  // Large DDC coordinates must not be narrowed to int, nor overflow
  // when translating an interval across its packed origin.
  using Index = ddc::DiscreteElementType;
  using Difference = ddc::DiscreteVectorElement;
  constexpr int maximum = std::numeric_limits<Difference>::digits;
  auto const spacing = Index{1} << (maximum - 3);
  auto parts = paliwa::pole_pass_segments(
      {static_cast<Difference>(spacing), static_cast<Difference>(5 * spacing)},
      6 * spacing, 3, maximum, 3);
  std::vector<std::array<int, 3>> actual;
  for (auto const &part : parts)
    for (int i = 0; i < part.count; ++i) {
      auto const offset = i * part.stride;
      actual.push_back(
          {part.first + offset, part.lower + offset, part.upper + offset});
    }
  EXPECT_EQ(actual,
            (std::vector<std::array<int, 3>>{{3, 2, 4}, {5, 4, 6}, {7, 6, 0}}));
}

TEST(transform_plan, packed_pass_index_range) {
  EXPECT_THROW(paliwa::pole_pass_segments({0, 0}, 0,
                                          std::numeric_limits<int>::digits,
                                          std::numeric_limits<int>::digits, 1),
               std::invalid_argument);
}
