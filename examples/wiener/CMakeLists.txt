cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

project(Wiener LANGUAGES CXX CUDA)

set(CMAKE_XXX_FLAGS "${CMAKE_XXX_FLAGS} ${QT5_EXECUTABLE_COMPILE_FLAGS}")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/../../cmake" )

find_package(BoltView REQUIRED)
find_package(Boost 1.53.0 COMPONENTS system filesystem program_options REQUIRED)
find_library(CUFFT NAME cufft PATHS /usr/local/cuda/lib64)

add_executable(wiener main.cu ../common/io.cc)
target_compile_definitions(wiener PRIVATE BOLT_USE_UNIFIED_MEMORY BOLT_ENABLE_FFT)
target_compile_features(wiener INTERFACE cxx_std_14)
target_include_directories(wiener PRIVATE ../common)
target_link_libraries(wiener Boost::program_options Boost::filesystem bolt)





