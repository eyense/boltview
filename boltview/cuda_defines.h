// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#pragma once

#ifdef __CUDACC__
#include <cuda.h>
#endif // __CUDACC__

#include <boost/format.hpp>
#include <boltview/debug.h>
#include <boltview/exceptions.h>
#include <string>


/// Macros for CUDA environment.
/// Error checking.
/// Wrappers for cuda keywords.


#if defined(__CUDACC__)
	inline std::ostream &operator<<(std::ostream &stream, const cudaPitchedPtr &pointer) {
		return stream << boost::format("[pitch: %1%; ptr %2%; xsize: %3%; ysize: %4%]")
			% pointer.pitch
			% pointer.ptr
			% pointer.xsize
			% pointer.ysize;
	}

	inline std::ostream &operator<<(std::ostream &stream, const cudaExtent &extent) {
		return stream << boost::format("[w: %1%; h %2%; d: %3%]")
			% extent.width
			% extent.height
			% extent.depth;
	}

	inline std::ostream &operator<<(std::ostream &stream, const dim3 &dimensions) {
		return stream << boost::format("[x: %1%; y %2%; z: %3%]")
			% dimensions.x
			% dimensions.y
			% dimensions.z;
	}

	inline std::string getErrorInfoStringForKernel(const std::string &kernel_name, dim3 grid_size, dim3 block_size) {
		return boost::str(boost::format("After %1% kernel, grid: %2%, block %3%") % kernel_name % grid_size % block_size);
	}


	#define BOLT_CHECK_MSG(error_message, ...) /* NOLINT */ \
		do {\
			cudaError_t err = __VA_ARGS__ ;\
			if(cudaSuccess != err) {\
				std::string msg = boost::str(boost::format("%1%:%2%: %3% %4%") % __FILE__ % __LINE__ % error_message % cudaGetErrorString(err));\
				BOLT_DFORMAT(msg); \
				BOLT_THROW(::bolt::CudaError() << ::bolt::MessageErrorInfo(msg));\
			}\
		} while (false);

	#define BOLT_CHECK(...) /* NOLINT */ \
		BOLT_CHECK_MSG(#__VA_ARGS__, __VA_ARGS__)

	#define BOLT_CHECK_ERROR_STATE(aErrorMessage) \
		BOLT_CHECK_MSG(aErrorMessage, cudaGetLastError());

	#define BOLT_CHECK_ERROR_AFTER_KERNEL(kernel, grid, block) \
		BOLT_CHECK_ERROR_STATE(getErrorInfoStringForKernel(kernel, grid, block));

	#define BOLT_DECL_HOST __host__
	#define BOLT_DECL_DEVICE __device__
	#define BOLT_DECL_HYBRID BOLT_DECL_HOST BOLT_DECL_DEVICE
	#define BOLT_GLOBAL __global__
	#define BOLT_CONSTANT __constant__
	#define BOLT_SHARED __shared__

	// Disables "host inside device function warning" when compiling with NVCC
	#if defined(__CUDACC__) && defined(__NVCC__)
		#if __CUDAVER__ >= 75000
			#define BOLT_HD_WARNING_DISABLE #pragma nv_exec_check_disable
		#else
			#define BOLT_HD_WARNING_DISABLE #pragma hd_warning_disable
		#endif
	#else
		#define BOLT_HD_WARNING_DISABLE
	#endif

#else
	// Only host and hybrid can be used in host only code and be still valid

	#define BOLT_DECL_HOST
	//#define BOLT_DECL_DEVICE
	#define BOLT_DECL_HYBRID
	//#define BOLT_GLOBAL
	//#define BOLT_CONSTANT
	//#define BOLT_SHARED

	#define BOLT_HD_WARNING_DISABLE
#endif  // __CUDACC__

//CUDA compiler and compilation side macros
#ifdef __NVCC__
	#define BOLT_NVCC
	#ifndef __CUDA_ARCH__
		#define BOLT_NVCC_HOST_CODE
		#define BOLT_HOST_CODE
 	#else
		#define BOLT_NVCC_KERNEL_CODE
		#define BOLT_KERNEL_CODE
 	#endif
#else
	#define BOLT_CLANG
	#ifndef __CUDA_ARCH__
		#define BOLT_CLANG_HOST_CODE
		#define BOLT_HOST_CODE
	#else
		#define BOLT_CLANG_KERNEL_CODE
		#define BOLT_KERNEL_CODE
	#endif
#endif

#ifdef BOLT_DISABLE_DEBUG_PRINTS

	#define BOLT_DFORMAT(...)  /* NOLINT */

#else
	#define BOLT_DFORMAT(...) D_FORMAT(__VA_ARGS__)  /* NOLINT */

#endif  // BOLT_DISABLE_DEBUG_PRINTS
// TODO(johny) - implement correct logging system
#define BOLT_ERROR_FORMAT(...) BOLT_DFORMAT(__VA_ARGS__)  /* NOLINT */

#if defined(CUDART_VERSION) && CUDART_VERSION <= 7050

	// NOTE: This is a workaround for a compile-time error when asserts are used in
	// hybrid class methods but compiles fine for asserts in hybrid functions,
	// See http://stackoverflow.com/questions/42541611/cannot-call-assert-from-inherited-class
	// Problem occurs with CUDA <=7.5, fixed in CUDA 8.0.
	BOLT_DECL_HYBRID inline void assert_fnc(bool val) {
		assert(val);
	}

	#define BOLT_ASSERT(...) /* NOLINT */ \
			assert_fnc(__VA_ARGS__);

#else

	#define BOLT_ASSERT(...) /* NOLINT */ \
			assert(__VA_ARGS__);

#endif
