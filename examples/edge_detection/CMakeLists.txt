cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

project(EdgeDetection LANGUAGES CXX CUDA)

set(CMAKE_XXX_FLAGS "${CMAKE_XXX_FLAGS} ${QT5_EXECUTABLE_COMPILE_FLAGS}")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/../../cmake" )

find_package(BoltView REQUIRED)
find_package(Boost 1.53.0 COMPONENTS system filesystem program_options REQUIRED)

add_executable(edge_detection main.cu ../common/io.cc)
target_compile_definitions(edge_detection PRIVATE BOLT_USE_UNIFIED_MEMORY)
target_compile_features(edge_detection INTERFACE cxx_std_14)
target_link_libraries(edge_detection Boost::program_options Boost::filesystem bolt)
target_include_directories(edge_detection PRIVATE ../common)





