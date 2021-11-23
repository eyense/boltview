// Copyright 2019 Eyen SE
// Author: Adam Kubista adam.kubista@eyen.se, Jan Kolomaznik jan.kolomaznik@eyen.se

#pragma once

#ifndef BOLT_ENABLE_FFT
#error "FFT is disabled in BoltView"
#endif

#include <complex>
#include <cufft.h>

#include <iostream>
#include <iomanip>


#include <boltview/device_image.h>
#include <boltview/device_image_view.h>
#include <boltview/host_image.h>

#include <boltview/math/complex.h>

namespace bolt {

/// \addtogroup FFT
/// @{

struct CuFFTError : CudaError {};
struct FFTWError : ExceptionBase {};


/// \brief Stacking policy
///
/// \tparam tDim1 -1 means no Stack, otherwise the dimension with the stack is in
/// \tparam tDim2 -1 means no Stack, this also allows for 2D stacks of 1D transforms
/// (f.e 0 means that data under indices (0, ...) and (1, ...) are from different inputs, but (0, ...) and (0, ...) are from the same input)
///
template<int tDim1, int tDim2 = -1>
struct Stack{
	static_assert((tDim1 == -1 && tDim2 == -1) || (tDim1 >= 0 && tDim1 <= 2 && tDim1 != tDim2 && tDim2 >= -1 && tDim2 <= 2), "Only X,Y,Z (0, 1, 2) respectively supported, also must not be same");
	// NOTE(fidi): this also denotes number of stacked dimensions
	static const int kIsStack = (tDim1 != -1 ? 1 : 0) + (tDim2 != -1 ? 1 : 0);
	static const int kDimension1 = tDim1;
	static const int kDimension2 = tDim2;
};

/// Returns the image dimensions for frequency spectrum
template <typename TStackPolicy = Stack<-1>>
BOLT_DECL_HYBRID inline Int3 getFftImageSize(Int3 size);

template <typename TStackPolicy = Stack<-1>>
BOLT_DECL_HYBRID inline Int2 getFftImageSize(Int2 size);

template <typename TStackPolicy = Stack<-1>>
BOLT_DECL_HYBRID inline Vector<int,1> getFftImageSize(int size);

template <typename TStackPolicy = Stack<-1>>
BOLT_DECL_HYBRID inline Vector<int,1> getFftImageSize(Vector<int,1> size);

/// Direction policy of FFT transform
struct Forward {};
/// Direction policy of FFT transform
struct Inverse {};

template<typename TSizeType>
struct DeviceFftPlan;

using HostNativePlanType = fftwf_plan;

/// Plan handle that runs on host, this should not be handled directly, use FftCalculator instead
template<typename TSizeType>
struct HostFftPlan{
	using NativePlanType = HostNativePlanType;
	struct FFTWDataDestroyer{
		void operator()(float * data) {
			fftwf_free(data);
		}
		void operator()(fftwf_complex * data) {
			fftwf_free(data);
		}
		void operator()(fftwf_plan * data) {
			fftwf_destroy_plan(*data);
		}
	};
	HostFftPlan() : plan(new HostNativePlanType) {};

	std::unique_ptr<HostNativePlanType, FFTWDataDestroyer> plan;
	TSizeType inputSize;
	TSizeType outputSize;
	std::unique_ptr<fftwf_complex[], FFTWDataDestroyer> frequencyDomainData;
	std::unique_ptr<float[], FFTWDataDestroyer> spaceDomainData;
	HostImageView<float, TSizeType::kDimension> spaceDomainView;
	HostImageView<HostComplexType, TSizeType::kDimension> frequencyDomainView;

	cudaStream_t stream = 0;
};

/// Library specific implementation API, this should not be handled directly, use FftCalculator instead
template<typename TDirection>
class HostFftPolicyHelper{
	public:
	template<typename TSizeType, typename TStackPolicy>
	static std::unique_ptr<HostFftPlan<TSizeType>> createPlan(TSizeType space_domain_size, TSizeType frequency_domain_size, TStackPolicy);
	template<typename TSizeType, typename TInputView, typename TOutputView>
	static void calculate(const HostFftPlan<TSizeType> * plan, TInputView input, TOutputView output);
	private:
	template<typename TSizeType, typename TInputView, typename TOutputView>
	static void requireDimensions(const HostFftPlan<TSizeType> * plan, TInputView input, TOutputView output);
};

/// \brief FftCalculator policy, This will run on host, single threaded.
///
/// \tparam TDirection direction of FFT, use Forward on Inverse
/// \tparam TStackPolicy stacking policy, leave out for single data FFT, otherwise use Stack
template<typename TDirection, typename TStackPolicy = Stack<-1>>
class HostFftPolicy{
	public:
		using StackPolicy = TStackPolicy;
		using Direction = TDirection;

		template<typename TSizeType>
		static std::unique_ptr<HostFftPlan<TSizeType>> createPlan(TSizeType space_domain_size, TSizeType frequency_domain_size){
			return HostFftPolicyHelper<TDirection>::createPlan(space_domain_size, frequency_domain_size, TStackPolicy());
		}
		template<typename TSizeType, typename TInputView, typename TOutputView>
		static void calculate(const HostFftPlan<TSizeType> * plan, TInputView input, TOutputView output){
			HostFftPolicyHelper<TDirection>::calculate(plan, input, output);
		}

	private:
};



template<typename TPolicy>
struct IsDevicePolicy : ::std::integral_constant<bool, false> {};

#if defined(__CUDACC__)
/// Library specific implementation API, this should not be handled directly, use FftCalculator instead
template<typename TDirection>
class DeviceFftPolicyHelper{
	public:
	template<typename TSizeType, typename TStackPolicy>
	static std::unique_ptr<DeviceFftPlan<TSizeType>> createPlan(TSizeType spaceDomainSize, TSizeType frequencyDomainSize, TStackPolicy);
	template<typename TSizeType, typename TInputView, typename TOutputView>
	static void calculateAsync(const DeviceFftPlan<TSizeType> * plan, TInputView input, TOutputView output);
	template<typename TSizeType, typename TStackPolicy>
	static uint64_t estimateWorkArea(const TSizeType spaceDomainSize, const TSizeType frequencyDomainSize, TStackPolicy);
	private:
	template<typename TSizeType, typename TInputView, typename TOutputView>
	static void requireDimensions(const DeviceFftPlan<TSizeType> * plan, TInputView input, TOutputView output);
};

/// \brief FftCalculator policy, This will run on device.
///
/// \tparam TDirection direction of FFT, use Forward on Inverse
/// \tparam TStackPolicy stacking policy, leave out for single data FFT, otherwise use Stack
template<typename TDirection, typename TStackPolicy = Stack<-1>>
class DeviceFftPolicy{
	public:
		using StackPolicy = TStackPolicy;
		using Direction = TDirection;

		template<typename TSizeType>
		static std::unique_ptr<DeviceFftPlan<TSizeType>> createPlan(TSizeType spaceDomainSize, TSizeType frequencyDomainSize){
			return DeviceFftPolicyHelper<TDirection>::createPlan(spaceDomainSize, frequencyDomainSize, TStackPolicy());
		}
		template<typename TSizeType, typename TInputView, typename TOutputView>
		static void calculateAsync(const DeviceFftPlan<TSizeType> * plan, TInputView input, TOutputView output){
			DeviceFftPolicyHelper<TDirection>::calculateAsync(plan, input, output);
		}

		template<typename TSizeType, typename TInputView, typename TOutputView>
		static void calculate(const DeviceFftPlan<TSizeType> * plan, TInputView input, TOutputView output){
			calculateAsync(plan, input, output);
			BOLT_CHECK(cudaStreamSynchronize(plan->stream));
		}

		template<typename TSizeType>
		static uint64_t estimateWorkArea(const TSizeType spaceDomainSize, const TSizeType frequencyDomainSize){
			return DeviceFftPolicyHelper<TDirection>::estimateWorkArea(spaceDomainSize, frequencyDomainSize, TStackPolicy());
		}
};

template<typename TDirection, typename TStackPolicy>
struct IsDevicePolicy<DeviceFftPolicy<TDirection, TStackPolicy>> : ::std::integral_constant<bool, true> {};


inline std::string getCuFFTErrorCodeMessage(cufftResult error_code) {
	static const std::map<cufftResult, std::string> kErrorMessages = {
		{ CUFFT_SUCCESS, "The cuFFT operation was successful" },
		{ CUFFT_INVALID_PLAN, "cuFFT was passed an invalid plan handle" },
		{ CUFFT_ALLOC_FAILED, "cuFFT failed to allocate GPU or CPU memory" },
		{ CUFFT_INVALID_TYPE, "No longer used" },
		{ CUFFT_INVALID_VALUE, "User specified an invalid pointer or parameter" },
		{ CUFFT_INTERNAL_ERROR, "Driver or internal cuFFT library error" },
		{ CUFFT_EXEC_FAILED, "Failed to execute an FFT on the GPU" },
		{ CUFFT_SETUP_FAILED, "The cuFFT library failed to initialize" },
		{ CUFFT_INVALID_SIZE, "User specified an invalid transform size" },
		{ CUFFT_UNALIGNED_DATA, "No longer used" },
		{ CUFFT_INCOMPLETE_PARAMETER_LIST, "Missing parameters in call" },
		{ CUFFT_INVALID_DEVICE, "Execution of a plan was on different GPU than plan creation" },
		{ CUFFT_PARSE_ERROR, "Internal plan database error " },
		{ CUFFT_NO_WORKSPACE, "No workspace has been provided prior to plan execution" },
		{ CUFFT_NOT_IMPLEMENTED, "Function does not implement functionality for parameters given" },
		{ CUFFT_LICENSE_ERROR, "Used in previous versions" },
		{ CUFFT_NOT_SUPPORTED, "Operation is not supported for parameters given" }
		};
	auto it = kErrorMessages.find(error_code);
	if (it == end(kErrorMessages)) {
		return "Unknown CUFFT error: " + std::to_string(error_code);
	}
	return it->second;
}



struct LineInfoTag{};
struct FileInfoTag{};

#define BOLT_CUFFT_CHECK_MSG(error_message, ...) \
		do {\
			cufftResult err = __VA_ARGS__ ;\
			if (CUFFT_SUCCESS  != err) {\
				std::string msg = boost::str(boost::format("%1% - %2% (%3%)") % error_message % bolt::getCuFFTErrorCodeMessage(err) % err);\
				BOLT_DFORMAT(msg); \
				auto e = ::bolt::CuFFTError(); \
				e << bolt::MessageErrorInfo(msg); \
				e << boost::error_info<::bolt::LineInfoTag,int>(__LINE__);\
				e << boost::error_info<::bolt::FileInfoTag,std::string>(__FILE__);\
				BOLT_THROW(e);\
			}\
		} while (false);

#define BOLT_CUFFT_CHECK(...) \
		BOLT_CUFFT_CHECK_MSG(#__VA_ARGS__, __VA_ARGS__)


using DeviceNativePlanType = cufftHandle;

/// Plan handle that runs on device, this should not be handled directly, use FftCalculator instead
template<typename TSizeType>
struct DeviceFftPlan{
	using NativePlanType = DeviceNativePlanType;
	struct CUFFTDataDestroyer{
		void operator()(cufftHandle * data) {
				if(data){
					cufftResult r = cufftDestroy(*data);
					if(r != CUFFT_SUCCESS && r != CUFFT_INVALID_PLAN){
						BOLT_CUFFT_CHECK(r);
					}
				}
		}
	};
	DeviceFftPlan() : plan(new DeviceNativePlanType) {};

	void setStream(cudaStream_t s){
		cufftSetStream(*plan, s);
		stream = s;
	}

	std::unique_ptr<DeviceNativePlanType, CUFFTDataDestroyer> plan;
	TSizeType inputSize;
	TSizeType outputSize;
	cudaStream_t stream = 0;
};

#endif

/// \brief Provides API for executing FFT
///
/// Usage: 1) create calculator by calling desired constructor of desired form
///        2) Perform calculate... methods on this object
///
/// \tparam tDim dimension of FFT transform
/// \tparam TPolicy DeviceFftPolicy or HostFftPolicy
///
/// E.g: FftCalculator<2,DeviceFftPolicy<Forward>> calculator(view.size());
///      Prepares calculator for 2D transform on device
///
///      FftCalculator<2,DeviceFftPolicy<Forward, Stack<0>>> calculator(view.size());
///      Prepares calculator for stack of 2D transforms on device, the individual inputs are stored in the 0th axis of input/output views
template <int tDim, typename TPolicy>
class FftCalculator {
	static_assert((tDim >= 1 && tDim <= 3), "Supporting only dimensions 1, 2, 3");
	public:
	using StackPolicy = typename TPolicy::StackPolicy;
	static const int kDataDim = tDim + StackPolicy::kIsStack;
	static_assert(kDataDim <= 3, "Can stack only 2D or 1D transforms in existent dimensions ( 2D in 3D etc...)");
	using SizeType = Vector<int, kDataDim>;
	static const int kDimension = tDim;
	using ComplexType = typename std::conditional<IsDevicePolicy<TPolicy>::value, DeviceComplexType, HostComplexType>::type;
	using PlanType = typename std::conditional<IsDevicePolicy<TPolicy>::value, DeviceFftPlan<SizeType>, HostFftPlan<SizeType>>::type;

	/// Attempts to create calculator according to given policies
	/// \param space_domain_size Always specify space domain size, including stack! (3d size, if stack of 2d transforms)
	explicit FftCalculator(SizeType space_domain_size) :
				space_domain_size_(space_domain_size),
				frequency_domain_size_(getFftImageSize<StackPolicy>(space_domain_size)){
		plan_ = TPolicy::createPlan(space_domain_size_, frequency_domain_size_);
	};


	FftCalculator(typename PlanType::NativePlanType * plan, SizeType space_domain_size) :
				plan_(new PlanType()),
				space_domain_size_(space_domain_size),
				frequency_domain_size_(getFftImageSize<StackPolicy>(space_domain_size)){
		static_assert(IsDevicePolicy<TPolicy>::value, "Must be device policy");
		// TODO(fidli): get rid of this constructor. Use different API in mgt to avoid such constructs...
		plan_->plan.reset(plan);
		if(::std::is_same<typename TPolicy::Direction, Forward>::value){
			plan_->inputSize = space_domain_size_;
			plan_->outputSize = frequency_domain_size_;
		}else {
			plan_->inputSize = frequency_domain_size_;
			plan_->outputSize = space_domain_size_;
		}
	};

	explicit FftCalculator(int space_domain_size) :
		space_domain_size_(Int1(space_domain_size)),
		frequency_domain_size_(getFftImageSize(space_domain_size)){
		static_assert(kDataDim == 1, "Only for dimension 1 and no stack");
		plan_ = TPolicy::createPlan(space_domain_size_, frequency_domain_size_);
	};

	// NOTE(fidli): mgt uses this, maybe re-consider if this makes sense:
	FftCalculator() : plan_(nullptr) {}

	SizeType getSpaceDomainSize(){
		return space_domain_size_;
	}

	SizeType getFrequencyDomainSize(){
		return frequency_domain_size_;
	}

#if defined(__CUDACC__)
	/// Creates frequency domain image of desired size for this calculator (including stacks)
	DeviceImage<ComplexType, kDataDim> createFrequencyDomainDeviceImage(){
		return DeviceImage<ComplexType, kDataDim>(frequency_domain_size_);
	}

	/// Creates space domain image of desired size for this calculator (including stacks)
	DeviceImage<float, kDataDim> createSpaceDomainDeviceImage(){
		return DeviceImage<float, kDataDim>(space_domain_size_);
	}
#endif  // defined(__CUDACC__)

	/// Creates frequency domain image of desired size for this calculator (including stacks)
	HostImage<ComplexType, kDataDim> createFrequencyDomainHostImage(){
		return HostImage<ComplexType, kDataDim>(frequency_domain_size_);
	}

	/// Creates space domain image of desired size for this calculator (including stacks)
	HostImage<float, kDataDim> createSpaceDomainHostImage(){
		return HostImage<float, kDataDim>(space_domain_size_);
	}

	/// Asynchronously calculates FFT according to calculator setup. Only for device policy
	template<typename TInputView, typename TOutputView>
	void calculateAsync(TInputView input, TOutputView output) const{
		static_assert(IsDevicePolicy<TPolicy>::value, "Async computation is available only for device.");
		TPolicy::calculateAsync(plan_.get(), input, output);
	}

	/// Synchronously calculates FFT according to calculator setup
	template<typename TInputView, typename TOutputView>
	void calculate(TInputView input, TOutputView output) const{
		TPolicy::calculate(plan_.get(), input, output);
	}

	/// \brief Synchronously calculates FFT according to calculator setup, and normalizes the output (considers stacks)
	template<typename TInputView, typename TOutputView>
	void calculateAndNormalize(TInputView input, TOutputView output) const{
		TPolicy::calculate(plan_.get(), input, output);
		normalizeInPlace(output, plan_->stream);
	}

	/// \brief Synchronously calculates FFT according to calculator setup, and normalizes the output (considers stacks)
	template<typename TInputView, typename TOutputView>
	void calculateAndNormalizeAsync(TInputView input, TOutputView output) const{
		TPolicy::calculateAsync(plan_.get(), input, output);
		normalizeInPlaceAsync(output, plan_->stream);
	}


	/// \brief Normalizes view by the space domain sample count (considers stacks too)
	template<typename TInputView>
	void normalizeInPlace(TInputView input, cudaStream_t stream = 0) const{
		copy(multiplyByFactor(1.0f / elementCount(space_domain_size_),  input), input, stream);
	}

	/// \brief Normalizes view by the space domain sample count (considers stacks too)
	template<typename TInputView>
	void normalizeInPlaceAsync(TInputView input, cudaStream_t stream = 0) const{
		copyAsync(multiplyByFactor(1.0f / elementCount(space_domain_size_),  input), input, stream);
	}

	/// \brief Estimates FFT work area according to calculator setup, Only for device policy
	static uint64_t estimateWorkArea(SizeType space_domain_size, SizeType frequency_domain_size){
		static_assert(IsDevicePolicy<TPolicy>::value, "Estimating work area is available only for device.");
		return TPolicy::estimateWorkArea(space_domain_size, frequency_domain_size);
	}

	uint64_t sizeInBytes() const{
		static_assert(IsDevicePolicy<TPolicy>::value, "Implement for host device");
		uint64_t size;
		BOLT_CUFFT_CHECK(cufftGetSize(*plan_->plan, &size));
		return size;
	}

	// NOTE(fidli): mgt uses this, what the fuck, it beats the purpouse
	// TODO(fidli): refactor, so this is not needed and fft calculator is used instead
	typename PlanType::NativePlanType * releaseNativePlan(){
		return plan_->plan.release();
	}

	void setStream(cudaStream_t stream){
		static_assert(IsDevicePolicy<TPolicy>::value, "Setting stream is available only for device.");
		plan_->setStream(stream);
	}

	protected:
	int64_t elementCount(SizeType size) const{
		int64_t result = 1;
		for(int i = 0; i < kDataDim; i++){
			if(i != StackPolicy::kDimension1 && i != StackPolicy::kDimension2){
				result *= get(size, i);
			}
		}
		return result;
	}
	SizeType space_domain_size_;
	SizeType frequency_domain_size_;
	std::unique_ptr<PlanType> plan_;

};

/// @}

}  // namespace bolt

#include <boltview/fft/fft_calculator.tcc>
