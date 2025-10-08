// SPDX-License-Identifier: MIT
// A lot reused from ddc/tests/strided_discrete_domain.cpp
// for these parts: Copyright (C) The DDC development team, see DDC's COPYRIGHT.md file

#include <algorithm>
#include <numeric>
#include <iostream>
#include <vector>

#include <ddc/ddc.hpp>
// #include <ddc/kernels/splines.hpp>

#include <Kokkos_Core.hpp>

struct X {};

// struct DDimX : ddc::UniformBSplines<X, 1>
// {
// };
struct DDimX : ddc::UniformPointSampling<X>
{
};
using DElemX = ddc::DiscreteElement<DDimX>;
using DVectX = ddc::DiscreteVector<DDimX>;
using DDomX = ddc::StridedDiscreteDomain<DDimX>;


struct Y {};
struct DDimY : ddc::UniformPointSampling<Y>
{
};
using DElemY = ddc::DiscreteElement<DDimY>;
using DVectY = ddc::DiscreteVector<DDimY>;
using DDomY = ddc::StridedDiscreteDomain<DDimY>;

struct Z {};
struct DDimZ : ddc::UniformPointSampling<Z>
{
};
using DElemZ = ddc::DiscreteElement<DDimZ>;
using DVectZ = ddc::DiscreteVector<DDimZ>;
using DDomZ = ddc::StridedDiscreteDomain<DDimZ>;

using DElemXYZ = ddc::DiscreteElement<DDimX, DDimY, DDimZ>;
using DVectXYZ = ddc::DiscreteVector<DDimX, DDimY, DDimZ>;
using DDomXYZ = ddc::DiscreteDomain<DDimX, DDimY, DDimZ>;
using SDDomXYZ = ddc::StridedDiscreteDomain<DDimX, DDimY, DDimZ>;

using DElemZYX = ddc::DiscreteElement<DDimZ, DDimY, DDimX>;
using DVectZYX = ddc::DiscreteVector<DDimZ, DDimY, DDimX>;
using DDomZYX = ddc::DiscreteDomain<DDimZ, DDimY, DDimX>;
using SDDomZYX = ddc::StridedDiscreteDomain<DDimZ, DDimY, DDimX>;

DElemX constexpr lbound_x = ddc::init_trivial_half_bounded_space<DDimX>();
DElemY constexpr lbound_y = ddc::init_trivial_half_bounded_space<DDimY>();
DElemZ constexpr lbound_z = ddc::init_trivial_half_bounded_space<DDimZ>();
DElemXYZ constexpr lbound_x_y_z(lbound_x, lbound_y, lbound_z);

double const x_start = 0.;
double const x_end = 1.;
double const y_start = 0.;
double const y_end = 1.;
double const z_start = 0.;
double const z_end = 1.;

SDDomXYZ strided_domain_from_level(std::vector<int> const& level, std::vector<int> const& finest_level) {
    std::vector<long int> resolution;
    std::ranges::transform(finest_level, std::back_inserter(resolution), [](int ml) { return (1 << ml) + 1; });
    DVectXYZ const resolution_xyz(resolution[0], resolution[1], resolution[2]);

    std::vector<int> const level_diff = {finest_level[0] - level[0],
                                         finest_level[1] - level[1],
                                         finest_level[2] - level[2]};
    std::vector<int> stride;
    std::ranges::transform(level_diff, std::back_inserter(stride), [](int l) { return (1 << l); });
    DVectXYZ const strides_xyz(stride[0], stride[1], stride[2]);
    return SDDomXYZ(lbound_x_y_z, resolution_xyz, strides_xyz);
}

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard const kokkos_scope;
    ddc::ScopeGuard const ddc_scope;

    std::vector<int> const maximum_level = {5, 6, 7};
    std::vector<long int> resolution;
    std::transform(
        maximum_level.begin(), maximum_level.end(), std::back_inserter(resolution), [](int ml) { return (1 << ml) + 1; });
    DVectXYZ const resolution_xyz(resolution[0], resolution[1], resolution[2]);
    // discrete domain in 3d, for the full grid but not allocated yet
    auto const x_domain_with_periodic_point = ddc::init_discrete_space<DDimX>(DDimX::init<DDimX>(
            ddc::Coordinate<X>(x_start),
            ddc::Coordinate<X>(x_end),
            ddc::DiscreteVector<DDimX>(resolution[0] + 1)));
    ddc::DiscreteDomain<DDimX> const x_domain
            = x_domain_with_periodic_point.remove_last(ddc::DiscreteVector<DDimX>(1));
    auto const y_domain_with_periodic_point = ddc::init_discrete_space<DDimY>(DDimY::init<DDimY>(
            ddc::Coordinate<Y>(y_start),
            ddc::Coordinate<Y>(y_end),
            ddc::DiscreteVector<DDimY>(resolution[1] + 1)));
    ddc::DiscreteDomain<DDimY> const y_domain
            = y_domain_with_periodic_point.remove_last(ddc::DiscreteVector<DDimY>(1));
    auto const z_domain_with_periodic_point = ddc::init_discrete_space<DDimZ>(DDimZ::init<DDimZ>(
            ddc::Coordinate<Z>(z_start),
            ddc::Coordinate<Z>(z_end),
            ddc::DiscreteVector<DDimZ>(resolution[2] + 1)));
    ddc::DiscreteDomain<DDimZ> const z_domain
            = z_domain_with_periodic_point.remove_last(ddc::DiscreteVector<DDimZ>(1));

    DDomXYZ const dom_x_y_z(lbound_x_y_z, resolution_xyz);
    

    std::vector<int> const minimum_level = {4, 5, 6};

    std::vector<std::vector<int>> all_levels = {{4,6,7}, {5,5,7}, {5,6,6}, {4,5,6}};
    std::vector<int> all_combi_coefficients = {1, 1, 1, -2};
    std::vector<SDDomXYZ> strided_domains;
    std::vector<ddc::Chunk<double, SDDomXYZ>> level_data;
    
    for (int grid_index = 0; grid_index < all_levels.size(); ++ grid_index){
        auto& level = all_levels[grid_index];
        auto& coefficient = all_combi_coefficients[grid_index];
        strided_domains.emplace_back(strided_domain_from_level(level, maximum_level));
        level_data.emplace_back(ddc::Chunk(
            "strided_grid_" + std::to_string(grid_index), strided_domains.back(),
            ddc::DeviceAllocator<double>()));
        auto strided_grid = level_data.back().span_view();
        
        // initialize!
        ddc::parallel_for_each(
            strided_domains.back(),
            KOKKOS_LAMBDA(DElemXYZ const ixyz) {
                double const x = ddc::coordinate(ddc::DiscreteElement<DDimX>(ixyz)); // ??
                double const y = ddc::coordinate(ddc::DiscreteElement<DDimY>(ixyz));
                double const z = ddc::coordinate(ddc::DiscreteElement<DDimZ>(ixyz));
                strided_grid(ixyz) = std::cos(3.0 + (x + y + z));
            });
        // ddc::parallel_fill(strided_grid, 2.5);

        //TODO how easiest for visualizable output? pdi? raw ofstream?
        // std::cout << strided_grid << std::endl; (-> issue)
        ddc::print_content(std::cout, strided_grid) << std::endl;
    }
}