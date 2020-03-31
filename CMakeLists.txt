cmake_minimum_required(VERSION 3.10)

project(BoltView CXX CUDA)


option(BOLT_ENABLE_FFT "Enable fft tools - add CUFFT as a dependency" ON)
option(BOLT_USE_UNIFIED_MEMORY "Enable features that use unified memory" OFF)


if(BOLT_USE_UNIFIED_MEMORY)
	add_definitions(-DBOLT_USE_UNIFIED_MEMORY)
endif()

option(BOLT_USE_THREAD_BUILDING_BLOCKS "Enable features that use thread building blocks" OFF)
if(BOLT_USE_THREAD_BUILDING_BLOCKS)
	add_definitions(-DBOLT_USE_THREAD_BUILDING_BLOCKS)
endif()

# Custom CMake commands
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# test if clang should be used as CUDA compiler
include(clang)
detect_clang_gpu()

include(test)

# # Find includes in corresponding build directories
# set(CMAKE_INCLUDE_CURRENT_DIR ON)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR})
#
# # C++ settings
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -Wall -ffast-math")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -fPIC -Wall -ffast-math")
#
# if (USE_CLANG_GPU)
# 	# disable some clang warnings
# 	set(CLANG_GPU_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-private-field -Wno-unused-local-typedef -Wno-unused-function")
# 	set(CLANG_GPU_C_FLAGS ${CMAKE_C_FLAGS})
# endif()
#
# Boost
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)

add_definitions(-DBOOST_ALL_NO_LIB)
find_package(Boost 1.53.0 COMPONENTS system filesystem REQUIRED)


# # Cuda
# #find_package(CUDA)
# if(NOT BOLT_GPU_ARCH)
# 	if("${CUDA_VERSION}" VERSION_GREATER 7.5)
# 		set(BOLT_GPU_ARCH "compute_61")
# 	else()
# 		set(BOLT_GPU_ARCH "compute_50")
# 	endif()
# endif()
# if(NOT BOLT_GPU_CODE)
# 	if("${CUDA_VERSION}" VERSION_GREATER 7.5)
# 		set(BOLT_GPU_CODE "sm_61")
# 	else()
# 		set(BOLT_GPU_CODE "sm_50")
# 	endif()
# endif()

# if (USE_CLANG_GPU)
# 	set(CLANG_GPU_CXX_FLAGS "${CLANG_GPU_CXX_FLAGS} --cuda-gpu-arch=${BOLT_GPU_CODE}")
# 	set(CLANG_GPU_C_FLAGS "${CLANG_GPU_C_FLAGS} --cuda-gpu-arch=${BOLT_GPU_CODE}")
# else()
# 	#set(NVCC_CPP11_FLAG "-std=c++11")
# 	#SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
# 	#SET(CUDA_GENCODE_FLAGS "-gencode arch=compute_30,code=sm_30 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=${BOLT_GPU_ARCH},code=${BOLT_GPU_CODE}")
# 	#set(CUDA_NVCC_FLAGS "${NVCC_CPP11_FLAG} -Xcudafe --display_error_number -Xcudafe --diag_suppress=2906 -Wno-deprecated-declarations -Xcompiler -fPIC ${CUDA_GENCODE_FLAGS} --use_fast_math   --expt-extended-lambda --generate-line-info") ## Add -gencode arch=compute_20,code=sm_20 etc. If you need other how support
# endif()

# set(CMAKE_CUDA_FLAGS "-Xcompiler -fPIC -gencode arch=${BOLT_GPU_ARCH},code=${BOLT_GPU_CODE} --use_fast_math   --expt-extended-lambda --generate-line-info" CACHE STRING "")

set(${PROJECT_NAME}_HEADERS
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/image_stack.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/region.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/device_image_view.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/procedural_views.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/view_traits.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/for_each.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/copy.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/texture_image.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/functors.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/math/complex.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/math/quaternion.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/math/vector.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/convolution_kernels.h"
	# "${CMAKE_CURRENT_SOURCE_DIR}/boltview/mgt/resource_constraints.h"
	# "${CMAKE_CURRENT_SOURCE_DIR}/boltview/mgt/scheduler_configuration.h"
	# "${CMAKE_CURRENT_SOURCE_DIR}/boltview/mgt/task_interface.h"
	# "${CMAKE_CURRENT_SOURCE_DIR}/boltview/mgt/task_base.h"
	# "${CMAKE_CURRENT_SOURCE_DIR}/boltview/mgt/device_code.h"
	# "${CMAKE_CURRENT_SOURCE_DIR}/boltview/mgt/memory_cache.h"
	# "${CMAKE_CURRENT_SOURCE_DIR}/boltview/mgt/gpu_worker.h"
	# "${CMAKE_CURRENT_SOURCE_DIR}/boltview/mgt/gpu_memory_manager.h"
	# "${CMAKE_CURRENT_SOURCE_DIR}/boltview/mgt/gpu_worker_configuration.h"
	# "${CMAKE_CURRENT_SOURCE_DIR}/boltview/mgt/image_pointers.h"
	# "${CMAKE_CURRENT_SOURCE_DIR}/boltview/mgt/scheduler.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/variadic_templates.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/interpolation.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/device_properties.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/array_view.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/kernel_execution_utils.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/host_image.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/filter.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/unified_image_view.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/device_future.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/image_io.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/fft/fft_calculator.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/fft/fft_views.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/fft/fft_utils.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/debug.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/view_iterators.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/execution_utils.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/transform.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/unified_image.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/texture_image_types.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/event.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/tuple.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/convolution.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/numeric_wrapper.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/device_image_view_base.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/image_view_utils.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/image_locator.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/unique_ptr.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/cuda_utils.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/host_image_view.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/subview.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/cuda_defines.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/stream.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/create_view.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/texture_image_view.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/gil_adaptor_image.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/shared_memory_static_array.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/reduce.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/int_sequence.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/move_utils.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/interpolated_view.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/geometrical_transformation.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/device_image.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/exception_error_info.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/fill.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/detail/convolution_foreach.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/detail/meta_algorithm_utils.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/detail/shared_memory_preload.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/detail/image_stack_reduce.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/detail/algorithm_common.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/host_image_view_base.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/view_policy.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/loop_utils.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/exceptions.h"
)

set(${PROJECT_NAME}_FFT_HEADERS
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/fft/fft_utils.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/fft/fft_views.h"
	)


set(${PROJECT_NAME}_TCC
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/image_io.tcc"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/reduce.tcc"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/math/complex.tcc"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/math/quaternion.tcc"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/math/vector.tcc"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/copy.tcc"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/mgt/gpu_memory_manager.tcc"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/texture_image_view.tcc"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/operators.tcc"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/cuda_utils.tcc"
)

set(${PROJECT_NAME}_TCC
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/fft/fft_utils.tcc"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/fft/fft_calculator.tcc"
	)

if(BOLT_ENABLE_FFT)
	set(${PROJECT_NAME}_HEADERS ${${PROJECT_NAME}_HEADERS} ${${PROJECT_NAME}_FFT_HEADERS})
	set(${PROJECT_NAME}_TCC ${${PROJECT_NAME}_TCC} ${${PROJECT_NAME}_FFT_TCC})
endif(BOLT_ENABLE_FFT)

add_library(bolt INTERFACE)
target_compile_features(bolt INTERFACE cxx_std_14)
target_sources(bolt INTERFACE $<BUILD_INTERFACE:${${PROJECT_NAME}_HEADERS}>)
target_link_libraries(bolt INTERFACE Boost::filesystem "$<$<BOOL:${BOLT_USE_THREAD_BUILDING_BLOCKS}>:tbb>")
target_include_directories(bolt INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_compile_options(bolt INTERFACE "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:--expt-extended-lambda>")


# Breakpoints
set(${PROJECT_NAME}_BPLIST
	${${PROJECT_NAME}_BPLIST}
	${${PROJECT_NAME}_HEADERS}
	${${PROJECT_NAME}_TCC}
)

option(BOLT_ENABLE_MGT "Compile module for multithreaded gpu usage." ON)

if (BOLT_ENABLE_MGT)
	add_subdirectory(boltview/mgt)
endif (BOLT_ENABLE_MGT)

include(breakpoints)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	breakpoints(${PROJECT_NAME}_BP "${PROJECT_BINARY_DIR}/breakpoints.gdb" ${PROJECT_NAME}_BPLIST)
endif()

add_subdirectory(tests)
# add_subdirectory(examples)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(PROJECT_CMAKE_CONFIG_DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/BoltViewConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/BoltViewConfig.cmake
    INSTALL_DESTINATION
        ${PROJECT_CMAKE_CONFIG_DESTINATION})


install(TARGETS bolt bolt_mgt
        PUBLIC_HEADER
        DESTINATION include
	)

# install(FILES 
# 	${${PROJECT_NAME}_HEADERS}
# 	${${PROJECT_NAME}_FFT_HEADERS}
# 	${${PROJECT_NAME}_TCC}
# 	DESTINATION include/boltview
# 	)

install(DIRECTORY
	boltview
	DESTINATION include
	)


install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/BoltViewConfig.cmake"
    DESTINATION
        ${PROJECT_CMAKE_CONFIG_DESTINATION})

# set(CPACK_GENERATOR TGZ;STGZ)
# set(CPACK_SOURCE_GENERATOR TGZ)

set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PACKAGE_CONTACT "jan.kolomaznik@eyen.se")
set(CPACK_PACKAGE_VENDOR "Eyen SE")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Jan Kolomaznik")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE.txt")
include(CPack)

