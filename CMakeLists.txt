cmake_minimum_required(VERSION 3.25)

# Set project language and version
project(distributed_wavelets_with_ddc
    LANGUAGES CXX
    VERSION 0.0.0)

set_property(GLOBAL PROPERTY GLOBAL_DEPENDS_NO_CYCLES ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra -fno-ipa-sra")

option(DDC_BUILD_BENCHMARKS       "Build DDC benchmarks." OFF)
option(DDC_BUILD_EXAMPLES         "Build DDC examples" OFF)
option(DDC_BUILD_TESTS            "Build DDC tests if BUILD_TESTING is enabled" OFF)

# follow spack env install instructions: https://ddc.mdls.fr/installation.html#autotoc_md35
find_package(DDC REQUIRED QUIET COMPONENTS fft pdi splines)
find_package(Kokkos REQUIRED)

add_executable( test_ddc_strided src/src.cpp )
target_link_libraries( test_ddc_strided PUBLIC DDC::core Kokkos::kokkos )
