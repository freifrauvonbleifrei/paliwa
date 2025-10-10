// SPDX-License-Identifier: MIT
// A lot reused from ddc/tests/strided_discrete_domain.cpp
// for these parts: Copyright (C) The DDC development team, see DDC's COPYRIGHT.md file

#include <algorithm>
#include <numeric>
#include <iostream>
#include <vector>

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

#include <Kokkos_Core.hpp>

#define PERIODIC_DOMAIN // Comment this to run non-periodic simulation

#define DIMENSIONALITY 2
static constexpr int8_t dimensionality = DIMENSIONALITY;

struct X {
#if defined(PERIODIC_DOMAIN)
    static constexpr bool PERIODIC = true;
#else
    static constexpr bool PERIODIC = false;
#endif
};
struct Y {
#if defined(PERIODIC_DOMAIN)
    static constexpr bool PERIODIC = true;
#else
    static constexpr bool PERIODIC = false;
#endif
};
#if DIMENSIONALITY > 2
struct Z {
#if defined(PERIODIC_DOMAIN)
    static constexpr bool PERIODIC = true;
#else
    static constexpr bool PERIODIC = false;
#endif
};
#endif

#if defined(PERIODIC_DOMAIN)
static constexpr ddc::BoundCond BoundCond = ddc::BoundCond::PERIODIC;
template <class DDim>
using ExtrapolationRule = ddc::PeriodicExtrapolationRule<DDim>;
#else
static constexpr ddc::BoundCond BoundCond = ddc::BoundCond::GREVILLE;
template <class DDim>
using ExtrapolationRule = ddc::NullExtrapolationRule;
#endif

template <class DDim>
using GrevillePoints = ddc::GrevilleInterpolationPoints<ddc::UniformBSplines<DDim, 1>, BoundCond, BoundCond>;
struct DDimX : GrevillePoints<X>::interpolation_discrete_dimension_type
{
};
// struct DDimX : ddc::UniformPointSampling<X>
// {
// };
using DElemX = ddc::DiscreteElement<DDimX>;
using DVectX = ddc::DiscreteVector<DDimX>;
using DDomX = ddc::StridedDiscreteDomain<DDimX>;


struct DDimY : GrevillePoints<Y>::interpolation_discrete_dimension_type
{
};
using DElemY = ddc::DiscreteElement<DDimY>;
using DVectY = ddc::DiscreteVector<DDimY>;
using DDomY = ddc::StridedDiscreteDomain<DDimY>;

#if DIMENSIONALITY > 2
struct DDimZ : GrevillePoints<Z>::interpolation_discrete_dimension_type
{
};
using DElemZ = ddc::DiscreteElement<DDimZ>;
using DVectZ = ddc::DiscreteVector<DDimZ>;
using DDomZ = ddc::StridedDiscreteDomain<DDimZ>;
#endif

using DElemXY = ddc::DiscreteElement<DDimX, DDimY>;
using DVectXY = ddc::DiscreteVector<DDimX, DDimY>;
using DDomXY = ddc::DiscreteDomain<DDimX, DDimY>;
using SDDomXY = ddc::StridedDiscreteDomain<DDimX, DDimY>;

#if DIMENSIONALITY > 2
using DElemXYZ = ddc::DiscreteElement<DDimX, DDimY, DDimZ>;
using DVectXYZ = ddc::DiscreteVector<DDimX, DDimY, DDimZ>;
using DDomXYZ = ddc::DiscreteDomain<DDimX, DDimY, DDimZ>;
using SDDomXYZ = ddc::StridedDiscreteDomain<DDimX, DDimY, DDimZ>;

using DElemZYX = ddc::DiscreteElement<DDimZ, DDimY, DDimX>;
using DVectZYX = ddc::DiscreteVector<DDimZ, DDimY, DDimX>;
using DDomZYX = ddc::DiscreteDomain<DDimZ, DDimY, DDimX>;
using SDDomZYX = ddc::StridedDiscreteDomain<DDimZ, DDimY, DDimX>;

using DElem = DElemXYZ;
using DVect = DVectXYZ;
using DDom = DDomXYZ;
using SDDom = SDDomXYZ;

// DElemZ constexpr lbound_z = ddc::init_trivial_half_bounded_space<DDimZ>();
// DElem constexpr lbound_all(lbound_x, lbound_y, lbound_z);
DElem constexpr lbound_all(0,0,0);

#else // DIMENSIONALITY > 2

using DElem = DElemXY;
using DVect = DVectXY;
using DDom = DDomXY;
using SDDom = SDDomXY;

// DElem constexpr lbound_all(lbound_x, lbound_y);
DElem constexpr lbound_all(0,0);

#endif // DIMENSIONALITY > 2

double const x_start = 0.;
double const x_end = 1.;
double const y_start = 0.;
double const y_end = 1.;
double const z_start = 0.;
double const z_end = 1.;

SDDom strided_domain_from_level(std::array<int, dimensionality> const& level, std::array<int, dimensionality> const& finest_level) {
    std::array<long int, dimensionality> resolution;
    std::ranges::transform(level, resolution.begin(), [](int ml) { return (1 << ml); });
    DVect resolution_all;
    ddc::detail::array(resolution_all) = resolution; //TODO temporary solution until assignment from std::array is implemented

    std::array<int, dimensionality> level_diff;
    std::ranges::transform(level, finest_level, level_diff.begin(), [](int l, int ml) { return ml - l; });
    std::array<long int, dimensionality> stride;
    std::ranges::transform(level_diff, stride.begin(), [](int l) { return (1 << l); });
    DVect strides_all;
    ddc::detail::array(strides_all) = stride; //TODO 
    return SDDom(lbound_all, resolution_all, strides_all);
}

template <typename ChunkType>
void dump_chunk_span_to_binary_file(ChunkType const span, std::string const& filename){
    std::ofstream file(filename, std::ios::trunc | std::ios::binary);
    auto chunk_size = span.size();
    for (auto i = 0; i < chunk_size; ++i) {
        file.write(reinterpret_cast<char*>(&span.data_handle()[i]), sizeof(span.data_handle()[i]));
    }
}

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard const kokkos_scope;
    ddc::ScopeGuard const ddc_scope;

#if DIMENSIONALITY > 2
    std::array<int, dimensionality> const maximum_level = {5, 6, 7};
#else
    std::array<int, dimensionality> const maximum_level = {4, 5};
#endif
    std::array<long int, dimensionality> resolution;
    std::transform(
        maximum_level.begin(), maximum_level.end(), resolution.begin(), [](int ml) { return (1 << ml) + 1; });
    DVect resolution_all;
    ddc::detail::array(resolution_all) = resolution; //TODO temporary solution until assignment from std::array is implemented
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
#if DIMENSIONALITY > 2
    auto const z_domain_with_periodic_point = ddc::init_discrete_space<DDimZ>(DDimZ::init<DDimZ>(
            ddc::Coordinate<Z>(z_start),
            ddc::Coordinate<Z>(z_end),
            ddc::DiscreteVector<DDimZ>(resolution[2] + 1)));
    ddc::DiscreteDomain<DDimZ> const z_domain
            = z_domain_with_periodic_point.remove_last(ddc::DiscreteVector<DDimZ>(1));

    DDom const dom_all(x_domain, y_domain, z_domain);
#else
    DDom const dom_all(x_domain, y_domain);
#endif

# if DIMENSIONALITY > 2
    std::array<int, dimensionality> const minimum_level = {4, 5, 6};
    std::vector<std::array<int, dimensionality>> all_levels = {{4,6,7}, {5,5,7}, {5,6,6}, {4,5,6}};
    std::vector<int> all_combi_coefficients = {1, 1, 1, -2};
# else
    std::array<int, dimensionality> const minimum_level = {2, 3};
    std::vector<std::array<int, dimensionality>> all_levels = {{2,5}, {3,4}, {4,3}, {2,4}, {3,3}};
    std::vector<int> all_combi_coefficients = {1, 1, 1, -1, -1};
# endif
    std::vector<SDDom> component_grid_domains;
    //TODO these as Kokkos unordered_map?
    std::vector<ddc::Chunk<double, SDDom>> level_data;
    
    for (int grid_index = 0; grid_index < all_levels.size(); ++ grid_index){
        auto& level = all_levels[grid_index];
        component_grid_domains.emplace_back(strided_domain_from_level(level, maximum_level));
        level_data.emplace_back(ddc::Chunk(
            "strided_grid_" + std::to_string(grid_index), component_grid_domains.back(),
            ddc::DeviceAllocator<double>()));
        auto strided_grid = level_data.back().span_view();
        
        // initialize!
        ddc::parallel_for_each(
            component_grid_domains.back(),
            KOKKOS_LAMBDA(DElem const ixyz) {
                double const x = ddc::coordinate(ddc::DiscreteElement<DDimX>(ixyz)); // ??
                double const y = ddc::coordinate(ddc::DiscreteElement<DDimY>(ixyz));
                #if DIMENSIONALITY > 2
                double const z = ddc::coordinate(ddc::DiscreteElement<DDimZ>(ixyz));
                strided_grid(ixyz) = std::cos(3.0 + (x + y + z));
                #else
                strided_grid(ixyz) = std::cos(3.0 + (x + y));
                #endif
            });

        //TODO how easiest for visualizable output? pdi? raw ofstream? (-> raw ofstream for now)
        std::string level_str = "";
        for (auto l : level) { level_str += std::to_string(l) + "_"; }
        level_str += std::to_string(dimensionality) + "d";
        std::string const filename = "strided_grid_" + level_str + ".raw";
        dump_chunk_span_to_binary_file(strided_grid, filename);
        // std::cout << strided_grid << std::endl; (-> issue)
        ddc::print_content(std::cout, strided_grid) << std::endl;
    }
}