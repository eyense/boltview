cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(Fractal LANGUAGES CXX CUDA)


find_package(BoltView REQUIRED)
find_package(Boost 1.53.0 COMPONENTS system filesystem program_options REQUIRED)

add_executable(fractal main.cu ../common/io.cc)
target_compile_definitions(fractal PRIVATE BOLT_USE_UNIFIED_MEMORY)
target_compile_features(fractal INTERFACE cxx_std_14)
target_include_directories(fractal PRIVATE ../common)
target_link_libraries(fractal Boost::program_options Boost::filesystem BoltView::bolt)





