cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

project(benchmarks CXX CUDA)

# Custom CMake commands
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

find_package(BoltView REQUIRED)
find_package(Boost 1.53.0 COMPONENTS system filesystem program_options timer REQUIRED)

add_executable(unified_memory_benchmarks unified_memory_benchmarks.cu benchmarks.cu benchmarks.h)
target_compile_definitions(unified_memory_benchmarks PRIVATE BOLT_USE_UNIFIED_MEMORY)
target_link_libraries(unified_memory_benchmarks Boost::program_options Boost::filesystem Boost::timer bolt)

add_executable(shared_memory shared_memory_benchmarks.cu benchmarks.cu benchmarks.h)
target_compile_definitions(shared_memory PRIVATE BOLT_USE_UNIFIED_MEMORY)
target_link_libraries(shared_memory Boost::program_options Boost::filesystem Boost::timer bolt)

add_executable(for_each_bench for_each_bench.cu benchmarks.cu benchmarks.h)
target_compile_definitions(for_each_bench PRIVATE BOLT_USE_UNIFIED_MEMORY)
target_link_libraries(for_each_bench Boost::program_options Boost::filesystem Boost::timer bolt)

add_executable(reduce_bench reduce_bench.cu benchmarks.cu benchmarks.h)
target_compile_definitions(reduce_bench PRIVATE BOLT_USE_UNIFIED_MEMORY)
target_link_libraries(reduce_bench Boost::program_options Boost::filesystem Boost::timer bolt)
# target_compile_options(reduce_bench PRIVATE "--generate-line-info")
target_compile_options(reduce_bench PRIVATE "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:--generate-line-info>")
