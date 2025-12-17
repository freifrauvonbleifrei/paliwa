cmake_minimum_required(VERSION 3.25)

# paliwa : parallel lifting wavelets
project(paliwa
    LANGUAGES CXX
    VERSION 0.0.0)

set_property(GLOBAL PROPERTY GLOBAL_DEPENDS_NO_CYCLES ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra -fno-ipa-sra")

option(PALIWA_BUILD_EXAMPLES         "Build paliwa examples" OFF)
option(PALIWA_BUILD_TESTS            "Build paliwa tests if BUILD_TESTING is enabled" OFF)
option(PALIWA_WITH_MPI               "Build paliwa with MPI support" ON)

include(CTest)

# follow spack env install instructions: https://ddc.mdls.fr/installation.html#autotoc_md35
find_package(DDC REQUIRED QUIET COMPONENTS fft pdi)
find_package(Kokkos REQUIRED COMPONENTS CXX)

if(PALIWA_WITH_MPI)
    find_package(MPI REQUIRED)
    message(STATUS "MPI support enabled")
    add_definitions(-DPALIWA_WITH_MPI)
else()
    message(STATUS "MPI support disabled")
endif()

add_executable( test_ddc_strided src/src.cpp )
target_link_libraries( test_ddc_strided PUBLIC DDC::core Kokkos::kokkos MPI::MPI_CXX )
#add_subdirectory(src) #TODO paliwa library & examples

if("${BUILD_TESTING}" AND "${PALIWA_BUILD_TESTS}")
    ## GoogleTest
    find_package(GTest REQUIRED) # version 1.14?
    add_subdirectory(tests/) #TODO tests
endif()