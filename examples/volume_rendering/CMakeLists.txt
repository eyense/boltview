cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

project(volume_rendering LANGUAGES CXX CUDA)

set(CMAKE_XXX_FLAGS "${CMAKE_XXX_FLAGS} ${QT5_EXECUTABLE_COMPILE_FLAGS}")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/../../cmake" )

find_package(BoltView REQUIRED)
find_package(Boost 1.53.0 COMPONENTS system filesystem program_options REQUIRED)

add_executable(volume_rendering main.cu ../common/io.cc)
target_compile_definitions(volume_rendering PRIVATE BOLT_USE_UNIFIED_MEMORY)
target_compile_features(volume_rendering INTERFACE cxx_std_14)
target_link_libraries(volume_rendering Boost::program_options Boost::filesystem bolt)
target_include_directories(volume_rendering PRIVATE ../common)





