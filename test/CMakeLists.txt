# SPDX-FileCopyrightText: 2025 The paliwa development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: LGPL-3.0-or-later

cmake_minimum_required(VERSION 3.25)

include(GoogleTest)


add_executable(
    paliwa_tests
    main.cpp
    test_paliwa_domains.cpp
    test_paliwa_restrict.cpp
    test_paliwa_transform.cpp
    test_paliwa_combination_technique.cpp
    test_paliwa_distributed.cpp
    )

target_compile_features(paliwa_tests PUBLIC cxx_std_17)
target_link_libraries(paliwa_tests PUBLIC GTest::gmock GTest::gtest DDC::core paliwa::core)
gtest_discover_tests(paliwa_tests DISCOVERY_MODE PRE_TEST)
