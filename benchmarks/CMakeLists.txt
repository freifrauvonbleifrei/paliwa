if(PALIWA_BUILD_BENCHMARKS)
    add_executable(
        paliwa_benchmarks_regression
        benchmark_main.cpp
        benchmark_common.hpp
        benchmark_regression.cpp
    )

    target_compile_features(paliwa_benchmarks_regression PUBLIC cxx_std_20)
    target_compile_definitions(paliwa_benchmarks_regression PRIVATE PALIWA_BENCHMARK_GENERAL)
    target_link_libraries(paliwa_benchmarks_regression PUBLIC benchmark::benchmark DDC::core paliwa::core)
    if(PALIWA_WITH_MPI)
        target_link_libraries(paliwa_benchmarks_regression PUBLIC MPI::MPI_CXX)
    endif()

    add_executable(
        paliwa_benchmarks_copy
        benchmark_main.cpp
        benchmark_common.hpp
        benchmark_copy.cpp
    )

    target_compile_features(paliwa_benchmarks_copy PUBLIC cxx_std_20)
    target_compile_definitions(paliwa_benchmarks_copy PRIVATE PALIWA_BENCHMARK_GENERAL)
    target_link_libraries(paliwa_benchmarks_copy PUBLIC benchmark::benchmark DDC::core paliwa::core)
    if(PALIWA_WITH_MPI)
        target_link_libraries(paliwa_benchmarks_copy PUBLIC MPI::MPI_CXX)
    endif()

    add_executable(
        paliwa_benchmarks_transform
        benchmark_main.cpp
        benchmark_common.hpp
        benchmark_transform.cpp
    )

    target_compile_features(paliwa_benchmarks_transform PUBLIC cxx_std_20)
    target_compile_definitions(paliwa_benchmarks_transform PRIVATE PALIWA_BENCHMARK_GENERAL)
    target_link_libraries(paliwa_benchmarks_transform PUBLIC benchmark::benchmark DDC::core paliwa::core)
    if(PALIWA_WITH_LIKWID)
        target_compile_definitions(paliwa_benchmarks_transform PRIVATE PALIWA_WITH_LIKWID LIKWID_PERFMON)
        target_include_directories(paliwa_benchmarks_transform PRIVATE ${LIKWID_INCLUDE_DIR})
        target_link_libraries(paliwa_benchmarks_transform PRIVATE ${LIKWID_LIBRARY})
    endif()
    if(PALIWA_WITH_MPI)
        target_link_libraries(paliwa_benchmarks_transform PUBLIC MPI::MPI_CXX)
    endif()

    add_executable(
        paliwa_benchmarks_distributed
        benchmark_main.cpp
        benchmark_common.hpp
        benchmark_distributed.cpp
    )

    target_compile_features(paliwa_benchmarks_distributed PUBLIC cxx_std_20)
    target_compile_definitions(paliwa_benchmarks_distributed PRIVATE PALIWA_BENCHMARK_GENERAL)
    target_link_libraries(paliwa_benchmarks_distributed PUBLIC benchmark::benchmark DDC::core paliwa::core)
    if(PALIWA_WITH_MPI)
        target_link_libraries(paliwa_benchmarks_distributed PUBLIC MPI::MPI_CXX)
    endif()
endif()

if(PALIWA_BUILD_BENCHMARKS AND PALIWA_WITH_MPI)
    add_executable(
        paliwa_benchmarks_strong
        benchmark_main.cpp
        benchmark_common.hpp
        benchmark_distributed.cpp
    )

    target_compile_features(paliwa_benchmarks_strong PUBLIC cxx_std_20)
    target_compile_definitions(paliwa_benchmarks_strong PRIVATE PALIWA_BENCHMARK_STRONG)
    target_link_libraries(paliwa_benchmarks_strong PUBLIC benchmark::benchmark DDC::core paliwa::core MPI::MPI_CXX)

    add_executable(
        paliwa_benchmarks_weak
        benchmark_main.cpp
        benchmark_common.hpp
        benchmark_distributed.cpp
    )

    target_compile_features(paliwa_benchmarks_weak PUBLIC cxx_std_20)
    target_compile_definitions(paliwa_benchmarks_weak PRIVATE PALIWA_BENCHMARK_WEAK)
    target_link_libraries(paliwa_benchmarks_weak PUBLIC benchmark::benchmark DDC::core paliwa::core MPI::MPI_CXX)
endif()
