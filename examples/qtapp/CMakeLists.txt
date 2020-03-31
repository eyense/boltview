cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

project(AnisotropicDiffusion LANGUAGES CXX CUDA)

set(CMAKE_XXX_FLAGS "${CMAKE_XXX_FLAGS} ${QT5_EXECUTABLE_COMPILE_FLAGS}")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/../cmake" )

find_package(BoltView)
find_package(Boost 1.53.0 COMPONENTS system filesystem program_options REQUIRED)
find_library(CUFFT NAME cufft PATHS /usr/local/cuda/lib64)

#add_executable(anisotropic_diffusion anisotropic_diffusion.cu)
#target_link_libraries(anisotropic_diffusion Boost::program_options Boost::filesystem ecip)
#target_compile_features(anisotropic_diffusion INTERFACE cxx_std_14)

find_package(Qt5 5.7.0 REQUIRED COMPONENTS Core Widgets)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie")

if ( TARGET Qt5::Core )
	get_property( core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS )
	string( REPLACE "-fPIC" "" new_core_options "${core_options}" )
	set_property( TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options} )
	set_property( TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON" )
	set(CMAKE_CXX_COMPILE_OPTIONS_PIE -fPIC )
	set(CMAKE_CUDA_COMPILE_OPTIONS_PIE -Xcompiler=-fPIC)

endif()
set_source_files_properties(${CU_SRCS} PROPERTIES COMPILE_FLAGS " -Xcompiler=-fPIC --use_fast_math ")
set_source_files_properties(${CXX_SRCS} PROPERTIES COMPILE_FLAGS " -fPIC --use_fast_math ")

add_executable(main main.cpp anisotropic.h anisotropic.cu mainwindow.h mainwindow.cpp mainwindow.ui diffusion_with_const_kernel.h diffusion_with_const_kernel.cu fileopen.h fileopen.cpp  thresholding.h thresholding.cu edge_detection.h edge_detection.cu  process_qimage.h process_qimage.cu  inverse_phase_correlation.h inverse_phase_correlation.cu phasecorrelation.h phasecorrelation.cpp phasecorrelation.ui phase_correlation_help.h phase_correlation_help.cu)
target_compile_definitions(main PRIVATE BOLT_USE_UNIFIED_MEMORY BOLT_ENABLE_FFT)
target_compile_features(main INTERFACE cxx_std_14)
target_link_libraries(main Boost::program_options Boost::filesystem Qt5::Core Qt5::Widgets ecip ${CUFFT} -lfftw3f -lm)





