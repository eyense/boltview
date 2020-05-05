cmake_minimum_required(VERSION 3.10)

project(BoltView 
	LANGUAGES CXX CUDA
	)

option(${PROJECT_NAME}_USE_THREAD_BUILDING_BLOCKS "Enable features that use thread building blocks" OFF)
option(${PROJECT_NAME}_ENABLE_FFT "Enable fft tools - add CUFFT as a dependency" ON)
option(${PROJECT_NAME}_USE_UNIFIED_MEMORY "Enable features that use unified memory" OFF)
option(${PROJECT_NAME}_ENABLE_TESTS "Enable compilation of tests" ON)

# Custom CMake commands
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# test if clang should be used as CUDA compiler
# include(clang)
# detect_clang_gpu()

include(test)

# Boost
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)

add_definitions(-DBOOST_ALL_NO_LIB)
find_package(Boost 1.53.0 COMPONENTS system filesystem REQUIRED)

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
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/mgt/gpu_memory_manager.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/mgt/gpu_worker.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/mgt/image_pointers.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/mgt/memory_cache.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/mgt/resource_constraints.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/mgt/scheduler_configuration.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/mgt/scheduler.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/mgt/task_base.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/mgt/task_interface.h"
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
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/fft/fft_utils.tcc"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/fft/fft_calculator.tcc"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/mgt/gpu_memory_manager.tcc"
	"${CMAKE_CURRENT_SOURCE_DIR}/boltview/mgt/scheduler.tcc"
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(bolt INTERFACE)
target_compile_features(bolt INTERFACE cxx_std_14)
target_sources(bolt INTERFACE $<BUILD_INTERFACE:${${PROJECT_NAME}_HEADERS}>)
target_link_libraries(bolt INTERFACE 
				Boost::filesystem 
				"$<$<BOOL:\$\{${PROJECT_NAME}_USE_THREAD_BUILDING_BLOCKS\}>:tbb>"
				)
target_include_directories(bolt INTERFACE 
				"$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
				"$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
				)
target_compile_options(bolt INTERFACE 
				"$<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>"
				)
target_compile_definitions(bolt INTERFACE
				"$<$<BOOL:\$\{${PROJECT_NAME}_USE_THREAD_BUILDING_BLOCKS\}>:BOLT_USE_THREAD_BUILDING_BLOCKS>"
				"$<$<BOOL:\$\{${PROJECT_NAME}_USE_UNIFIED_MEMORY\}>:BOLT_USE_UNIFIED_MEMORY>"
				)


# Breakpoints
set(${PROJECT_NAME}_BPLIST
	${${PROJECT_NAME}_BPLIST}
	${${PROJECT_NAME}_HEADERS}
	${${PROJECT_NAME}_TCC}
)


include(breakpoints)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	breakpoints(${PROJECT_NAME}_BP "${PROJECT_BINARY_DIR}/breakpoints.gdb" ${PROJECT_NAME}_BPLIST)
endif()

if(${PROJECT_NAME}_ENABLE_TESTS)
	add_subdirectory(tests)
endif()
# add_subdirectory(examples)
set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "0")

write_basic_package_version_file("${PROJECT_NAME}ConfigVersion.cmake"
	VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}"
                                 COMPATIBILITY SameMajorVersion)
configure_package_config_file(
				"${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
				"${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
				INSTALL_DESTINATION
				${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

install(TARGETS bolt EXPORT ${PROJECT_NAME}_Targets)
install(EXPORT ${PROJECT_NAME}_Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)
install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
              "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)
install(DIRECTORY boltview
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
	)

# install(FILES 
# 	${${PROJECT_NAME}_HEADERS}
# 	${${PROJECT_NAME}_FFT_HEADERS}
# 	${${PROJECT_NAME}_TCC}
# 	DESTINATION include/boltview
# 	)



# install(
#     FILES
#         "${CMAKE_CURRENT_BINARY_DIR}/BoltViewConfig.cmake"
#     DESTINATION
#         ${PROJECT_CMAKE_CONFIG_DESTINATION})

set(CPACK_GENERATOR TGZ;STGZ;DEB)
# set(CPACK_SOURCE_GENERATOR TGZ)

set(CPACK_PACKAGE_CONTACT "jan.kolomaznik@eyen.se")
set(CPACK_PACKAGE_VENDOR "Eyen SE")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Jan Kolomaznik")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE.txt")
include(CPack)

