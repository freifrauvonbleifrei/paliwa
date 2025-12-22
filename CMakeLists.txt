# SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: LGPL-2.1-or-later

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

include(CMakePackageConfigHelpers)
include(CTest)
include(GNUInstallDirs)

# follow spack env install instructions: https://ddc.mdls.fr/installation.html#autotoc_md35
find_package(DDC REQUIRED QUIET) # COMPONENTS fft pdi)
find_package(Kokkos REQUIRED COMPONENTS CXX)

if(PALIWA_WITH_MPI)
    find_package(MPI REQUIRED)
    message(STATUS "MPI support enabled")
    add_definitions(-DPALIWA_WITH_MPI)
else()
    message(STATUS "MPI support disabled")
endif()

## paliwa core library
add_library(paliwa_core INTERFACE)
add_library(paliwa::core ALIAS paliwa_core)
set_target_properties(paliwa_core PROPERTIES EXPORT_NAME core)
# configure_file(cmake/config.hpp.in generated/ddc/config.hpp NO_SOURCE_PERMISSIONS @ONLY)
target_sources(
    paliwa_core
    INTERFACE
        FILE_SET HEADERS
            #BASE_DIRS include ${CMAKE_CURRENT_BINARY_DIR}/generated
            FILES
                #${CMAKE_CURRENT_BINARY_DIR}/generated/ddc/config.hpp
                src/paliwa_dimensions.hpp
                src/paliwa_distribute.hpp
                src/paliwa_domains.hpp
                src/paliwa_io.hpp
                src/paliwa_transform.hpp
                src/paliwa_utils.hpp
                src/paliwa_wavelets.hpp
)
target_compile_features(paliwa_core INTERFACE cxx_std_20)
target_link_libraries(paliwa_core INTERFACE Kokkos::kokkos)
target_link_libraries(paliwa_core INTERFACE DDC::core)
if(PALIWA_WITH_MPI)
    target_link_libraries(paliwa_core INTERFACE MPI::MPI_CXX)
    target_compile_definitions(paliwa_core INTERFACE PALIWA_WITH_MPI)
endif()

install(TARGETS paliwa_core EXPORT PaliwaCoreTargets FILE_SET HEADERS)

add_executable( test_ddc_strided src/src.cpp )
target_link_libraries( test_ddc_strided PUBLIC paliwa::core DDC::core Kokkos::kokkos )
if(PALIWA_WITH_MPI)
    target_link_libraries( test_ddc_strided PUBLIC MPI::MPI_CXX )
    target_compile_definitions( test_ddc_strided PUBLIC PALIWA_WITH_MPI )
endif()

if("${BUILD_TESTING}" AND "${PALIWA_BUILD_TESTS}")
    ## GoogleTest
    find_package(GTest REQUIRED) # version 1.14?
    add_subdirectory(test/)
endif()


## installation

install(EXPORT PaliwaCoreTargets NAMESPACE paliwa:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/paliwa)

configure_package_config_file(
    cmake/PaliwaConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/PaliwaConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/paliwa
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# We use SameMinorVersion until major version 1
if(PROJECT_VERSION VERSION_GREATER_EQUAL 1)
    message(FATAL_ERROR "must switch COMPATIBILITY mode to SameMajorVersion")
endif()
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/PaliwaConfigVersion.cmake
    COMPATIBILITY SameMinorVersion
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/PaliwaConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/PaliwaConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/paliwa
)