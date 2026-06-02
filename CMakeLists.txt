# SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Some parts reused from ddc/CMakeLists.txt
# for these parts:
# Copyright (C) The DDC development team, see DDC's COPYRIGHT.md file

# SPDX-License-Identifier: MIT

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
option(PALIWA_BUILD_BENCHMARKS       "Build Google Benchmark executables" OFF)
option(PALIWA_WITH_MPI               "Build paliwa with MPI support" ON)
option(PALIWA_WITH_LIKWID            "Enable LIKWID marker support in the transform benchmark" OFF)

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

if(PALIWA_WITH_LIKWID)
    find_program(SPACK_EXECUTABLE spack)
    set(LIKWID_PREFIX "")
    if(SPACK_EXECUTABLE)
        execute_process(
            COMMAND "${SPACK_EXECUTABLE}" location -i likwid
            OUTPUT_VARIABLE LIKWID_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE LIKWID_PREFIX_RESULT)
    endif()

    if(NOT LIKWID_PREFIX_RESULT EQUAL 0 OR NOT EXISTS "${LIKWID_PREFIX}")
        if(DEFINED ENV{LIKWID_ROOT})
            set(LIKWID_PREFIX "$ENV{LIKWID_ROOT}")
        elseif(DEFINED LIKWID_ROOT)
            set(LIKWID_PREFIX "${LIKWID_ROOT}")
        endif()
    endif()

    find_path(LIKWID_INCLUDE_DIR
        NAMES likwid-marker.h
        HINTS "${LIKWID_PREFIX}"
        PATH_SUFFIXES include)
    find_library(LIKWID_LIBRARY
        NAMES likwid
        HINTS "${LIKWID_PREFIX}"
        PATH_SUFFIXES lib)

    if(NOT LIKWID_INCLUDE_DIR OR NOT LIKWID_LIBRARY)
        message(FATAL_ERROR "Could not find LIKWID headers/libraries. Set LIKWID_ROOT or install likwid via Spack.")
    endif()

    message(STATUS "LIKWID support enabled: ${LIKWID_PREFIX}")
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
                paliwa/paliwa_dimensions.hpp
                paliwa/paliwa_distribute.hpp
                paliwa/paliwa_distributed_transform.hpp
                paliwa/paliwa_domains.hpp
                paliwa/paliwa_io.hpp
                paliwa/paliwa_transform.hpp
                paliwa/paliwa_utils.hpp
                paliwa/paliwa_wavelets.hpp
)
target_compile_features(paliwa_core INTERFACE cxx_std_20)
target_link_libraries(paliwa_core INTERFACE Kokkos::kokkos)
target_link_libraries(paliwa_core INTERFACE DDC::core)
if(PALIWA_WITH_MPI)
    target_link_libraries(paliwa_core INTERFACE MPI::MPI_CXX)
    target_compile_definitions(paliwa_core INTERFACE PALIWA_WITH_MPI)
endif()

install(TARGETS paliwa_core EXPORT PaliwaCoreTargets FILE_SET HEADERS)

## no example, todo
# add_executable( test_ddc_strided paliwa/src.cpp )
# target_link_libraries( test_ddc_strided PUBLIC paliwa::core DDC::core Kokkos::kokkos )
# if(PALIWA_WITH_MPI)
#     target_link_libraries( test_ddc_strided PUBLIC MPI::MPI_CXX )
#     target_compile_definitions( test_ddc_strided PUBLIC PALIWA_WITH_MPI )
# endif()

if(("${BUILD_TESTING}" AND "${PALIWA_BUILD_TESTS}") OR "${PALIWA_BUILD_BENCHMARKS}")
    ## GoogleTest
    if("${BUILD_TESTING}" AND "${PALIWA_BUILD_TESTS}")
        find_package(GTest QUIET) # version 1.14?
        if(NOT GTest_FOUND)
            if(EXISTS "/usr/src/googletest/CMakeLists.txt")
                message(STATUS "Using local googletest sources from /usr/src/googletest")
                set(BUILD_GMOCK ON CACHE BOOL "Build GoogleMock" FORCE)
                set(INSTALL_GTEST OFF CACHE BOOL "Do not install GoogleTest" FORCE)
                add_subdirectory(/usr/src/googletest ${CMAKE_BINARY_DIR}/googletest EXCLUDE_FROM_ALL)
            else()
                message(FATAL_ERROR "Could not find GTest or local googletest sources")
            endif()
        endif()
    endif()

    if("${PALIWA_BUILD_BENCHMARKS}")
        find_package(benchmark QUIET)
        if(NOT benchmark_FOUND)
            message(FATAL_ERROR "Could not find Google Benchmark package 'benchmark'")
        endif()
    endif()

    if("${BUILD_TESTING}" AND "${PALIWA_BUILD_TESTS}")
        add_subdirectory(test/)
    endif()
    if("${PALIWA_BUILD_BENCHMARKS}")
        add_subdirectory(benchmarks/)
    endif()
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