cmake_minimum_required(VERSION 3.10)

project(BoltViewMGT CXX CUDA)

add_definitions(
	-DBOOST_ALL_NO_LIB 
	-DBOOST_RESULT_OF_USE_DECLTYPE # so we can pass lambdas to utilities like transform_iterator
	) 
find_package(Boost 1.53.0 REQUIRED)
find_library(CUFFT_LIB cufft /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu/)


set(BOLT_MGT_HEADERS 
	"${CMAKE_CURRENT_SOURCE_DIR}/gpu_memory_manager.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/gpu_memory_manager.tcc"
	"${CMAKE_CURRENT_SOURCE_DIR}/gpu_worker.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/image_pointers.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/memory_cache.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/resource_constraints.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/scheduler.tcc"
	"${CMAKE_CURRENT_SOURCE_DIR}/scheduler_configuration.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/scheduler.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/task_base.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/task_interface.h"
		)

add_library(bolt_mgt INTERFACE)
target_compile_features(bolt_mgt INTERFACE cxx_std_14)
target_sources(bolt_mgt INTERFACE $<BUILD_INTERFACE:${BOLT_MGT_HEADERS}>)
target_compile_options(bolt_mgt INTERFACE "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:--expt-extended-lambda>")
target_link_libraries(bolt_mgt INTERFACE bolt ${CUFFT_LIB})

