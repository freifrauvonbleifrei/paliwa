// SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md
//
// SPDX-License-Identifier: LGPL-3.0-or-later

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
          ddc::host_for_each(
              domain, [&](Element i) { grid(i) = input[i.uid<PlanDim>()]; });
          paliwa::transform_in<PlanDim>(grid, Vector(level), Vector(maximum),
                                        levels, filters, outputs, exec);
          exec.fence();
          ddc::host_for_each(outputs, [&](Element i) {
            EXPECT_NEAR(grid(i), reference[i.uid<PlanDim>()], 1e-12);
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
