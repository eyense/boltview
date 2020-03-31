cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

project(Fractal LANGUAGES CXX CUDA)

set(CMAKE_XXX_FLAGS "${CMAKE_XXX_FLAGS} ${QT5_EXECUTABLE_COMPILE_FLAGS}")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/../../cmake" )

find_package(BoltView REQUIRED)
find_package(Boost 1.53.0 COMPONENTS system filesystem program_options REQUIRED)
find_library(CUFFT NAME cufft PATHS /usr/local/cuda/lib64)

add_executable(fractal main.cu ../common/io.cc)
target_compile_definitions(fractal PRIVATE BOLT_USE_UNIFIED_MEMORY)
target_compile_features(fractal INTERFACE cxx_std_14)
target_include_directories(fractal PRIVATE ../common)
target_link_libraries(fractal Boost::program_options Boost::filesystem bolt)
set_target_properties(fractal PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(fractal PROPERTIES POSITION_INDEPENDENT_CODE ON)





