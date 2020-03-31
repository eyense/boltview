// Copyright 2017 Eyen SE
// Author: Jan Cerveny jan.cerveny@eyen.se

#pragma once

#include <boltview/math/vector.h>
#include <boltview/image_locator.h>
#include <boltview/loop_utils.h>
#include <boltview/transform.h>
#include <boltview/convolution_kernels.h>


namespace bolt {


/// Functor that holds convolution kernel and returns result of convolution
template<typename TConvolutionKernel, typename TOutputElement>
struct ConvolutionFunctor{
	TConvolutionKernel kernel_;

	BOLT_DECL_HYBRID
	explicit ConvolutionFunctor(TConvolutionKernel kernel):
		kernel_(kernel)
	{}

	template<typename TLocator>
	BOLT_DECL_HYBRID
	TOutputElement operator()(TLocator locator) const {
		TOutputElement sum = 0;

			sum = sumEachNeighbor(
				kernelStart(kernel_),
				kernelEnd(kernel_),
				(TOutputElement)0,
				[&](Vector<int, TConvolutionKernel::kDimension> index){
					return locator[index] * kernel_[index];
				});
			return sum;
		}
};


/// Returns kernel view - kernel class without destructor
template<typename TConvolutionKernel, bool tIsDynamicallyAllocated>
struct KernelView{
	using Type = typename TConvolutionKernel::Predecessor;

	static typename TConvolutionKernel::Predecessor get(const TConvolutionKernel & kernel){
		return static_cast<typename TConvolutionKernel::Predecessor>(kernel);
	}
};

/// Template specialization for static kernels, just return itself
template<typename TConvolutionKernel>
struct KernelView<TConvolutionKernel, false>{
	using Type = TConvolutionKernel;

	static TConvolutionKernel get(const TConvolutionKernel & kernel){
		return kernel;
	}
};

/// \return Coordinates of beginning of the kernel, relative to kernel center
template<typename TConvolutionKernel>
BOLT_DECL_HYBRID
Vector<int, TConvolutionKernel::kDimension> kernelStart(const TConvolutionKernel & kernel){
	return -kernel.center();
}


/// \return Coordinates 1 behind of end of the kernel, relative to kernel center
template<typename TConvolutionKernel>
BOLT_DECL_HYBRID
Vector<int, TConvolutionKernel::kDimension> kernelEnd(const TConvolutionKernel & kernel){
	return kernel.size() - kernel.center();
}


/// \return index to 1-dimensional array from given coordinates
template<typename TConvolutionKernel, typename TIndex>
BOLT_DECL_HYBRID
int64_t kernelIndex(const TConvolutionKernel & kernel, TIndex index){
	return getLinearAccessIndex(kernel.size(), index+kernel.center());
}


/// Convolves view with convolution kernel
/// \param view_in Input view, can be read only
/// \param view_out Output view - must provide write access
/// \param convolution_kernel convolution kernel to be applied
/// \param policy Policy class describing kernel execution configuration.
/// \param cuda_stream Which stream should schedule this operation
template<typename TInView, typename TOutView, typename TConvolutionKernel, typename TPolicy>
void convolution(TInView view_in, TOutView view_out, const TConvolutionKernel & convolution_kernel, TPolicy policy, cudaStream_t cuda_stream = nullptr){
	const bool views_host =  TInView::kIsHostView && TOutView::kIsHostView;
	const bool views_device =  TInView::kIsDeviceView && TOutView::kIsDeviceView;

	static_assert(views_host || views_device,
		"Incompatible views. Views have to be both device, both host or at least one of them has to be UnifiedImageView");

	const bool kernel_host_only = TConvolutionKernel::kIsHostKernel && !TConvolutionKernel::kIsDeviceKernel;
	const bool kernel_device_only = TConvolutionKernel::kIsDeviceKernel && !TConvolutionKernel::kIsHostKernel;

	static_assert(!(kernel_device_only && !views_device), "Cannot use device-only kernel on non-device views");
	static_assert(!(kernel_host_only && !views_host), "Cannot use host-only kernel on non-host views");

	if(TPolicy::kPreloadToSharedMemory){
		policy.setPreload(convolution_kernel.size(), convolution_kernel.center());
	}

	transformLocator(
			view_in,
			view_out,
			ConvolutionFunctor<
				typename KernelView<TConvolutionKernel, TConvolutionKernel::kIsDynamicallyAllocated>::Type,
				typename TOutView::Element>
					(KernelView<TConvolutionKernel, TConvolutionKernel::kIsDynamicallyAllocated>::get(convolution_kernel)),
			policy,
			cuda_stream);
}

/// Convolves view with convolution kernel
/// \param view_in Input view, can be read only
/// \param view_out Output view - must provide write access
/// \param convolution_kernel convolution kernel to be applied
/// \param cuda_stream Which stream should schedule this operation
template<typename TInView, typename TOutView, typename TConvolutionKernel>
void convolution(TInView view_in, TOutView view_out, const TConvolutionKernel & convolution_kernel, cudaStream_t cuda_stream = nullptr){
	convolution(view_in, view_out, convolution_kernel, DefaultTransformLocatorPolicy<TInView, TOutView>(), cuda_stream);
}


namespace detail{

template<int tDimension>
struct SeparableConvolutionImplementation {
	template<typename TInView, typename TOutView, typename  TTmpView, typename TSeparableKernel, typename TPolicy>
	static void run(TInView view_in, TOutView view_out, TTmpView view_tmp, const TSeparableKernel & kernel, TPolicy policy, cudaStream_t cuda_stream){

		auto kernel0 = kernel.get(0);
		if(TPolicy::kPreloadToSharedMemory){
			policy.setPreload(kernel0.size(), kernel0.center());
		}
		convolution(constView(view_in), view_tmp, kernel0, policy, cuda_stream);

		auto kernel1 = kernel.get(1);
		if(TPolicy::kPreloadToSharedMemory){
			policy.setPreload(kernel1.size(), kernel1.center());
		}
		convolution(constView(view_tmp), view_out, kernel1, policy, cuda_stream);
	}
};

template<>
struct SeparableConvolutionImplementation<3> {
	template<typename TInView, typename TOutView, typename  TTmpView, typename TSeparableKernel, typename TPolicy>
	static void run(TInView view_in, TOutView view_out, TTmpView view_tmp, const TSeparableKernel & kernel, TPolicy policy, cudaStream_t cuda_stream){
		auto kernel0 = kernel.get(0);
		if(TPolicy::kPreloadToSharedMemory){
			policy.setPreload(kernel0.size(), kernel0.center());
		}
		convolution(constView(view_in), view_out, kernel0, policy, cuda_stream);

		auto kernel1 = kernel.get(1);
		if(TPolicy::kPreloadToSharedMemory){
			policy.setPreload(kernel1.size(), kernel1.center());
		}
		convolution(constView(view_out), view_tmp, kernel1, policy, cuda_stream);

		auto kernel2 = kernel.get(2);
		if(TPolicy::kPreloadToSharedMemory){
			policy.setPreload(kernel2.size(), kernel2.center());
		}
		convolution(constView(view_tmp), view_out, kernel2, policy, cuda_stream);
	}
};


}  // namespace detail


/// Convolves (separable) view with convolution kernel
/// \param view_in Input view, can be read only
/// \param view_out Output view - must provide write access
/// \param view_tmp Temporary view - must provide write access
/// \param convolution_kernel convolution kernel to be applied
/// \param policy Policy class describing kernel execution configuration.
/// \param cuda_stream Which stream should schedule this operation
template<typename TInView, typename TOutView, typename  TTmpView, typename TSeparableKernel, typename TPolicy>
void separableConvolution(TInView view_in, TOutView view_out, TTmpView view_tmp, const TSeparableKernel & convolution_kernel, TPolicy policy, cudaStream_t cuda_stream = nullptr){
	static_assert(TInView::kDimension == TSeparableKernel::kDimension, "View and kernel have different dimensions.");
	detail::SeparableConvolutionImplementation<TSeparableKernel::kDimension>::run(view_in,
		view_out,
		view_tmp,
		convolution_kernel,
		policy,
		cuda_stream);
}

/// Convolves (separable) view with convolution kernel
/// \param view_in Input view, can be read only
/// \param view_out Output view - must provide write access
/// \param view_tmp Temporary view - must provide write access
/// \param convolution_kernel convolution kernel to be applied
/// \param cuda_stream Which stream should schedule this operation
template<typename TInView, typename TOutView, typename  TTmpView, typename TSeparableKernel>
void separableConvolution(TInView view_in, TOutView view_out, TTmpView view_tmp, const TSeparableKernel & convolution_kernel, cudaStream_t cuda_stream = nullptr){
		separableConvolution(view_in, view_out, view_tmp, convolution_kernel, DefaultTransformLocatorPolicy<TInView, TOutView>(), cuda_stream);
}


namespace detail{
// From am_devlib convolution
constexpr double constSqrt(int value) {
	return value == 1 ? 1 : 1.4142135623730951;
}

/// Generator for 1-dimensional gaussian that can store the data in generic container
template<typename TStorage>
class HybridGaussGenerator{
public:
	using Element = typename TStorage::value_type;

	/// \param std_dev Standart deviation
	explicit HybridGaussGenerator(Element variance):
		variance_(variance),
		size_(static_cast<int>(4*constSqrt(variance) + 0.5)*2+1),
		kernel_(size_/2+1)
	{
		computeKernel();
	}
	/// \params size Size of 'image'
	/// \params numStdDevAtSize standard deviation of the Gauss function is computed such that the edge of the
	/// image (i.e. at Size/2) has the value of std_dev * numStdDevAtSize. The higher this number, the "sharper"
	/// the gauss peak will appear within the image. The constructor taking only std_dev is equivalent to this one with
	/// numStdDevAtSize = 2 and size = (int)(std_dev * 2) * 2 + 1
	explicit HybridGaussGenerator(int size, float num_variance_at_size):
		variance_((size / 2) / num_variance_at_size),
		size_(size),
		kernel_(size_/2+1)
	{
		computeKernel();
	}

	Element get(int idx){
		if(idx > size_/2){
			idx = size_ - idx - 1;
		}
		return kernel_[idx];
	}

	int size() const{
		return size_;
	}

private:
	int size_;
	Element variance_;
	TStorage kernel_;

	void computeKernel(){
		Element sum = 0;

		// Count half of the gaussian
		for(int i = -size_/2; i <= 0; ++i){
				Element res = exp(-((Element)i*i)/(2.0*variance_));
				kernel_[i+size_/2] = res;
				sum += res;
		}

		sum = 2*sum - 1;  // Substract center of the gaussian that was counted twice
		// normalization
		for(int i = 0; i < size_/2+1; ++i){
			kernel_[i] /= sum;
		}
	}
};

template<typename TType>
using GaussGenerator = HybridGaussGenerator<std::vector<TType>>;

}  // namespace detail

#ifdef BOLT_USE_UNIFIED_MEMORY

// TODO(honza) asymmetric separable gaussians

/// \return SeparableKernel filled with gaussian
/// \param std_dev Standart deviation
template <int tDimension>
SeparableKernel<float, tDimension> getSeparableGaussian(float std_dev){
	detail::GaussGenerator<float> generator(std_dev);
	Vector<int, tDimension> center;
	Vector<int, tDimension> size;

	for(int i = 0; i < tDimension; ++i){
		size[i] = generator.size();
		center[i] = size[i] / 2;
	}

	std::unique_ptr<float[]> array(new float[sum(size)]);

	for(int i = 0; i < tDimension; ++i){
		for(int j = 0; j < generator.size(); ++j){
			array[i*generator.size() + j] = generator.get(j);
		}
	}

	return SeparableKernel<float, tDimension>(array.get(), size, center);
}


/// \return SeparableKernel filled with asymmetric 2D gaussian
/// \param std_dev Standart deviation
template <int tDimension> inline
DynamicUnifiedKernel<float, tDimension> getGaussian(float std_devX, float std_devY){
	static_assert(tDimension == 2, "1D not implemented yet.");

	detail::GaussGenerator<float> generatorX(std_devX);
	detail::GaussGenerator<float> generatorY(std_devY);

	Vector<int, tDimension> size_all;
	Vector<int, tDimension> center;

	size_all[0] = generatorX.size();
	size_all[1] = generatorY.size();

	center[0] = size_all[0] / 2;
	center[1] = size_all[1] / 2;

	std::unique_ptr<float[]> array(new float[product(size_all)]);

	for(int i = 0; i < size_all[1]; ++i){
		for(int j = 0; j < size_all[0]; ++j){
			float res = generatorY.get(i) * generatorX.get(j);
			array[i*size_all[0] + j] = res;
		}
	}

	return DynamicUnifiedKernel<float, tDimension>(size_all, center, array.get());
}


template <int tDimension>
DynamicUnifiedKernel<float, tDimension> getGaussian(float std_dev){
	return getGaussian<tDimension>(std_dev, std_dev);
}

// TODO(honza) 3D asymmetric gaussian and sobel
//             Edge detection using sobel

template <>
inline DynamicUnifiedKernel<float, 3> getGaussian<3>(float std_dev){
	detail::GaussGenerator<float> generator(std_dev);
	int size = generator.size();
	Vector<int, 3> size_all;
	Vector<int, 3> center;

	for(int i = 0; i < 3; ++i){
		size_all[i] = size;
		center[i] = size / 2;
	}

	std::unique_ptr<float[]> array(new float[product(size_all)]);

	float sum = 0;

	for(int i = 0; i < size; ++i){
		for(int j = 0; j < size; ++j){
			for(int k = 0; k < size; ++k){
				float res = generator.get(i) * generator.get(j) * generator.get(k);
				array[i*size*size + j*size + k] = res;
				sum += res;
			}
		}
	}

	return DynamicUnifiedKernel<float, 3>(size_all, center, array.get());
}


/// \return SeparableKernel filled with sobel for x-direction
inline SeparableKernel<int, 2> getSeparableSobelX(){
	int kernel[] = {1, 0, -1, 1, 2, 1};

	Int2 size(3, 3);
	Int2 center(1, 1);

	return SeparableKernel<int, 2>(kernel, size, center);
}


/// \return SeparableKernel filled with sobel for y-direction
inline SeparableKernel<int, 2> getSeparableSobelY(){
	int kernel[] = {1, 2, 1, 1, 0, -1};

	Int2 size(3, 3);
	Int2 center(1, 1);

	return SeparableKernel<int, 2>(kernel, size, center);
}


inline int getSobelValue(int position, bool shift){
	int values[] = {1, 2, 1,  1, 0, -1};
	return values[position+3*shift];
}


/// \return Unified memory kernel filled with sobel for given axis
template<int tDimension, int tAxis>
DynamicUnifiedKernel<int, tDimension> getSobel(){
	static_assert(tAxis < tDimension,	"Axis value cannot be greater than dimension");

	int sobel[27];
	int values[3];

	int endY = 3;
	int endZ = 3;

	if(tDimension < 2){
		endY = 1;
	}
	if(tDimension < 3){
		endZ = 1;
	}

	for(int z = 0; z < endZ; ++z){
		values[2] = getSobelValue(z, tAxis == 2);

		for(int y = 0; y < endY; ++y){
			values[1] = getSobelValue(y, tAxis == 1);

			for(int x = 0; x < 3; ++x){
				values[0] = getSobelValue(x, tAxis == 0);

				int r = values[0] * values[1] * values[2];
				sobel[x+3*y+9*z] = r;
			}
		}
	}

	auto size = Vector<int, tDimension>::Fill(3);
	auto center = Vector<int, tDimension>::Fill(1);

	return DynamicUnifiedKernel<int, tDimension>(size, center, sobel);
}

#endif // BOLT_USE_UNIFIED_MEMORY

}  // namespace bolt
