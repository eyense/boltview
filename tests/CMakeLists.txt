cmake_minimum_required(VERSION 3.12)

project(BoltView_tests CXX CUDA)


option(BoltView_ENABLE_AUTOMATED_TESTS "Enable running automated tests using 'make test'" ON)
if (BoltView_ENABLE_AUTOMATED_TESTS)
	enable_testing()
endif()


include(test)

# Tests
set(${PROJECT_NAME}_TESTS
	"operators_test"
	"quaternion_test"
	"vector_test"
	"view_test"
	"interpolation_test"
	"image_locator_test"
	"algorithm_test"
	"array_view_test"
	#"texture_image_test" is added later if USE_CLANG_GPU is not used
	"unique_ptr_test"
	"image_test"
	"image_io_test"
	"convolution_test"
	"reduce_test"
	"interpolated_view_test"
	"view_iterators_test"
	"device_future_test"
	"gil_adaptor_image_test"
	"test_64bit_indices"
	# "image_stack_test"
)
if(BoltView_ENABLE_FFT)
	find_library(CUFFT_LIB NAME cufft PATHS /usr/local/cuda/lib64)
	list( APPEND ${PROJECT_NAME}_TESTS "fft_utils_test")
endif(BoltView_ENABLE_FFT)

bolt_test_suite("." ${PROJECT_NAME}_TESTS "")
# add_subdirectory(mgt)

if(BoltView_ENABLE_FFT)
	target_link_libraries(fft_utils_test ${CUFFT_LIB} -lfftw3f -lm)
endif(BoltView_ENABLE_FFT)

if (!USE_CLANG_GPU)
	# there were problems with cudaTextureObject_t type in clang
	list( APPEND ${PROJECT_NAME}_TESTS "texture_image_test")
endif()

target_link_libraries(device_future_test pthread)


add_executable(coding_style EXCLUDE_FROM_ALL coding_style.cc)
target_link_libraries(coding_style bolt)
target_clang_check(coding_style ".*")


