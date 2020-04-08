// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#pragma once

#include <limits>
#include <cmath>
#include <cfloat>

#include <thrust/functional.h>

#include <boltview/array_view.h>
#include <boltview/create_view.h>
#include <boltview/subview.h>

#if defined(__CUDACC__)
	#include <boltview/cuda_defines.h>

	#include <thrust/system/cuda/execution_policy.h>
#endif  // __CUDACC__

#include <boltview/procedural_views.h>
#include <boltview/view_iterators.h>
//#include <boltview/shared_memory_static_array.h>

#ifdef __CUDACC__
	#include <boltview/device_image.h>
#endif // __CUDACC__

namespace bolt {

namespace detail {

#ifdef __CUDACC__

template <int tInputDimension, int tOutputDimension, int tDirection, int tBlockSize>
struct IndexTraits;

template <int tDirection, int tBlockSize>
struct IndexTraits<2, 1, tDirection, tBlockSize> {
	BOLT_DECL_DEVICE
	static Int2 getInputIndex() {
		auto tmp = Vector<int, 1>(blockIdx.y);
		return insertDimension(tmp, int(blockIdx.x * (tBlockSize * 2) + threadIdx.x), tDirection);
	}

	BOLT_DECL_DEVICE
	static int getOutputIndex(Int2 index) {
		return removeDimension(index, tDirection)[0];
	}
};

template <int tDirection, int tBlockSize>
struct IndexTraits<2, 2, tDirection, tBlockSize> {
	BOLT_DECL_DEVICE
	static Int2 getInputIndex() {
		auto tmp = Vector<int, 1>(blockIdx.y);
		return insertDimension(tmp, int(blockIdx.x * (tBlockSize * 2) + threadIdx.x), tDirection);
	}

	BOLT_DECL_DEVICE
	static Int2 getOutputIndex(Int2 index) {
		auto tmp = Vector<int, 1>(blockIdx.y);
		return insertDimension(tmp, int(blockIdx.x), tDirection);
	}
};

template <int tDirection, int tBlockSize>
struct IndexTraits<3, 2, tDirection, tBlockSize> {
	BOLT_DECL_DEVICE
	static Int3 getInputIndex() {
		auto tmp = Int2(blockIdx.y, blockIdx.z);
		return insertDimension(tmp, int(blockIdx.x * (tBlockSize * 2) + threadIdx.x), tDirection);
	}

	BOLT_DECL_DEVICE
	static Int2 getOutputIndex(Int3 index) {
		return removeDimension(index, tDirection);
	}
};

template <int tDirection, int tBlockSize>
struct IndexTraits<3, 3, tDirection, tBlockSize> {
	BOLT_DECL_DEVICE
	static Int3 getInputIndex() {
		auto tmp = Int2(blockIdx.y, blockIdx.z);
		return insertDimension(tmp, int(blockIdx.x * (tBlockSize * 2) + threadIdx.x), tDirection);
	}

	BOLT_DECL_DEVICE
	static Int3 getOutputIndex(Int3 index) {
		auto tmp = Int2(blockIdx.y, blockIdx.z);
		return insertDimension(tmp, int(blockIdx.x), tDirection);
	}
};


/// Based on <a href="https://docs.nvidia.com/cuda/samples/6_Advanced/reduction/doc/reduction.pdf">Optimizing Parallel Reduction in CUDA by Mark Harris</a>
template <typename TView, typename TOutputView, typename TOutputValue, typename TOperator, int tDimension, int tBlockSize>
BOLT_GLOBAL void dimensionReduceKernel(
	TView view,
	TOutputView output_view,
	TOutputValue initial_value,
	TOperator reduction_operator)
{
	using Indexing = IndexTraits<TView::kDimension, TOutputView::kDimension, tDimension, tBlockSize>;
	__shared__ TOutputValue sdata[tBlockSize];
	__syncthreads();  // Wait for all threads to call constructor on sdata
	//SharedMemoryStaticArray<TOutputValue, tBlockSize, true> sdata;

	int tid = threadIdx.x;
	int count_in_dimension = view.size()[tDimension];
	// sdata[tid] = initial_value;

	__syncthreads();
	auto index = Indexing::getInputIndex();
	auto output_index = Indexing::getOutputIndex(index);
	int step = tBlockSize * 2 * gridDim.x;

	{
		TOutputValue tmp_value = initial_value;
		while (index[tDimension] < count_in_dimension - tBlockSize) {
			auto other_index = index;
			other_index[tDimension] += tBlockSize;
			tmp_value = reduction_operator(tmp_value, reduction_operator(view[index], view[other_index]));
			index[tDimension] += step;
		}
		__syncthreads();
		if (index[tDimension] < count_in_dimension) {
			tmp_value = reduction_operator(tmp_value, view[index]);
		}
		sdata[tid] = tmp_value;
	}
	__syncthreads();
	if (tBlockSize >= 512) {
		if (tid < 256) {
			sdata[tid] = reduction_operator(sdata[tid], sdata[tid + 256]);
		}
		__syncthreads();
	}
	if (tBlockSize >= 256) {
		if (tid < 128) {
			sdata[tid] = reduction_operator(sdata[tid], sdata[tid + 128]);
		}
		__syncthreads();
	}
	if (tBlockSize >= 128) {
		if (tid < 64) {
			sdata[tid] = reduction_operator(sdata[tid], sdata[tid + 64]);
		}
	}
	__syncthreads();
	for (int i = 32; i > 0; i >>= 1) {
		if (tid < i) {
			sdata[tid] = reduction_operator(sdata[tid], sdata[tid + i]);
		}
		__syncthreads();
	}
	if (tid == 0) {
		output_view[output_index] = sdata[0];
	}


}

template <typename TView, typename TOperator, typename TOutputValue, typename TOutputView, int tBlockSize>
BOLT_GLOBAL void reduceKernel(
	TView view,
	TOutputValue initial_value,
	TOperator reduction_operator,
	TOutputView output
	)
{
	static_assert(tBlockSize >= 64, "Block must have at least 64 threads.");
	__shared__ TOutputValue sdata[tBlockSize];
	__syncthreads();  // Wait for all threads to call constructor on sdata
	//SharedMemoryStaticArray<TOutputValue, tBlockSize, true> sdata;
	// TODO: we still have warnings about __shared__ memory variable with non-empty constructor -> perhaps a wrapper?

	int tid = threadIdx.x;
	int index = blockIdx.x * (tBlockSize * 2) + tid;
	int grid_size = tBlockSize * 2 * gridDim.x;
	sdata[tid] = initial_value;
	int element_count = view.elementCount();
	while (index < element_count - tBlockSize) {
		sdata[tid] = reduction_operator(sdata[tid], reduction_operator(linearAccess(view, index), linearAccess(view, index + tBlockSize)));
		index += grid_size;
	}
	__syncthreads();
	if (index < element_count) {
		sdata[tid] = reduction_operator(sdata[tid], linearAccess(view, index));
	}
	__syncthreads();
	if (tBlockSize >= 512) {
		if (tid < 256) {
			sdata[tid] = reduction_operator(sdata[tid], sdata[tid + 256]);
		}
		__syncthreads();
	}
	if (tBlockSize >= 256) {
		if (tid < 128) {
			sdata[tid] = reduction_operator(sdata[tid], sdata[tid + 128]);
		}
		__syncthreads();
	}
	if (tBlockSize >= 128) {
		if (tid < 64) {
			sdata[tid] = reduction_operator(sdata[tid], sdata[tid + 64]);
		}
	}
	__syncthreads();
	for (int i = 32; i > 0; i >>= 1) {
		if (tid < i) {
			sdata[tid] = reduction_operator(sdata[tid], sdata[tid + i]);
		}
		__syncthreads();
	}
	if (tid == 0) {
		output[blockIdx.x] = sdata[0];
	}
}

#endif // __CUDACC__


template<typename TSize>
int estimateCurrentLevelTemporary(TSize size, ExecutionPolicy execution_policy) {
	auto block_size = execution_policy.block_size;
	return 1 + (product(size) -1) / (block_size * execution_policy.bucket_size);
}

template<typename TSize>
int estimateTemporaryBuffer(TSize size, ExecutionPolicy execution_policy) {
	auto current_level = estimateCurrentLevelTemporary(size, execution_policy);
	if (current_level == 1) {
		return current_level;
	} else {
		return current_level + estimateTemporaryBuffer(current_level, execution_policy);
	}
}



template<bool tRunOnDevice>
struct ReduceImplementation {
#ifdef __CUDACC__
	template<typename TView, typename TmpView, typename TOutputValue, typename TOperator>
	static void run(TView inview, TmpView tmp_view, TOutputValue initial_value, TOperator reduction_operator, ExecutionPolicy execution_policy) {
		int tmp_size = detail::estimateTemporaryBuffer(inview.size(), execution_policy);
		int current_size = detail::estimateCurrentLevelTemporary(inview.size(), execution_policy);
		int following_layers = tmp_size - current_size;

		auto current_tmp = subview(tmp_view, following_layers, current_size);
		auto following_tmp = subview(tmp_view, 0, following_layers);


		dim3 block(execution_policy.block_size, 1, 1);
		dim3 grid(current_size, 1, 1);

		reduceKernel<TView, TOperator, TOutputValue, decltype(current_tmp), 512><<<grid, block, 0, execution_policy.cuda_stream>>>(inview, initial_value, reduction_operator, current_tmp);
		BOLT_CHECK_ERROR_AFTER_KERNEL("reduceKernel", grid, block);
		if (current_tmp.size() > 1) {
			run<decltype(current_tmp), decltype(following_tmp), TOutputValue, TOperator>(current_tmp, following_tmp, initial_value, reduction_operator, execution_policy);
		}
	}

	/*template<typename TView, typename TOutputValue, typename TOperator>
	static void run(TView inview, TOutputValue &output_value, TOutputValue initial_value, TOperator reduction_operator, ExecutionPolicy execution_policy) {
		const int kBucketSize = execution_policy.bucket_size;  // Bundle more computation in one block - prevents thread idling. TODO(johny) - should be specified by call policy.
		dim3 block(execution_policy.block_size, 1, 1);
		dim3 grid(1 + (inview.elementCount() - 1) / (block.x * kBucketSize), 1, 1);

		// thrust::device_vector<TOutputValue> tmp_vector(grid.x);
		DeviceImage<TOutputValue, 1> tmp_vector(grid.x);
		// tmp_vector.clear();
		auto tmp = view(tmp_vector);
		reduceKernel<TView, TOperator, TOutputValue, decltype(tmp), 512><<<grid, block, 0, execution_policy.cuda_stream>>>(inview, initial_value, reduction_operator, tmp);
		BOLT_CHECK_ERROR_AFTER_KERNEL("reduceKernel", grid, block);
		if (grid.x > 1) {
			// auto tmp = makeDeviceArrayView(tmp_vector);
			reduceAsync(tmp, output_value, initial_value, reduction_operator, execution_policy);
		} else {
			tmp.getOnHost(0, output_value, execution_policy.cuda_stream);
		}
	}*/

	template<typename TView, typename TOutputView, typename TOutputValue, int tDimension, typename TOperator>
	static void run(TView inview, TOutputView output_view, DimensionValue<tDimension> dimension, TOutputValue initial_value, TOperator reduction_operator, ExecutionPolicy execution_policy) {
		auto size = inview.size();
		auto reduced_size = removeDimension(size, tDimension);

		if (reduced_size != output_view.size()) {
			BOLT_THROW(IncompatibleViewSizes() << getViewPairSizesErrorInfo(reduced_size, output_view.size()));
		}

		// constexpr int kBlockSize = 256;
		constexpr int kBlockSize = 128;
		constexpr int kBucketSize = 2;
		int block_count = 1 + (size[tDimension] - 1) / (kBlockSize * kBucketSize);
		// TODO - better setup of bucket size and block size depending on input size
		if (block_count > 1) {
			dim3 block(kBlockSize, 1, 1);
			dim3 grid(block_count, reduced_size[0], reduced_size.kDimension >= 2 ? reduced_size[1] : 1);

			auto tmp_size = size;
			tmp_size[tDimension] = block_count;
			DeviceImage<TOutputValue, TView::kDimension> tmp_buffer(tmp_size);
			dimensionReduceKernel<TView, decltype(tmp_buffer.view()), TOutputValue, TOperator, tDimension, kBlockSize><<<grid, block, 0, execution_policy.cuda_stream>>>(inview, tmp_buffer.view(), initial_value, reduction_operator);
			BOLT_CHECK_ERROR_AFTER_KERNEL("dimensionReduceKernel tmp buffer", grid, block);
			run(tmp_buffer.constView(), output_view, dimension, initial_value, reduction_operator, execution_policy);
		} else {
			dim3 block(kBlockSize, 1, 1);
			dim3 grid(1 + (size[tDimension] - 1) / block.x, reduced_size[0], reduced_size.kDimension >= 2 ? reduced_size[1] : 1);

			dimensionReduceKernel<TView, TOutputView, TOutputValue, TOperator, tDimension, kBlockSize><<<grid, block, 0, execution_policy.cuda_stream>>>(inview, output_view, initial_value, reduction_operator);
			BOLT_CHECK_ERROR_AFTER_KERNEL("dimensionReduceKernel", grid, block);
			BOLT_CHECK(cudaStreamSynchronize(execution_policy.cuda_stream));
		}
	}
#endif // __CUDACC__
};


template<>
struct ReduceImplementation<false> {
	template<typename TView, typename TOutputValue, typename TOperator>
	static void run(TView inview, TOutputValue &output_value, TOutputValue initial_value, TOperator reduction_operator, ExecutionPolicy /*execution_policy*/) {
		// TODO(johny) - execution policy is not used
		TOutputValue result = initial_value;
		for (int i = 0; i < inview.elementCount(); ++i) {
			result = reduction_operator(result, linearAccess(inview, i));  // TODO(johny) more efficient traversal
		}
		output_value = result;
	}

	template<typename TView, typename TOutputView, typename TOutputValue, int tDimension, typename TOperator>
	static void run(TView inview, TOutputView output_view, DimensionValue<tDimension>  /*dimension*/, TOutputValue initial_value, TOperator reduction_operator, ExecutionPolicy /*execution_policy*/) {
		// TODO(johny) - execution policy is not used
		for (auto& value : output_view){
			value = initial_value;
		}
		for (auto pair : zipWithPosition(inview)){
			auto reduced_position = removeDimension(pair.position, tDimension);
			output_view[reduced_position] = reduction_operator(output_view[reduced_position], inview[pair.position]);
		}
	}
};

}  // namespace detail



template<typename TOutputValue, bool tRunOnDevice>
class Reduce {
public:

	template<typename TSize>
	Reduce(TSize size, ExecutionPolicy execution_policy = ExecutionPolicy{}) :
		execution_policy_(execution_policy)
	{}

	template<typename TView, typename TOperator>
	void runAsync(TView input, TOutputValue &output_value, TOutputValue initial_value, TOperator reduction_operator) {
		detail::ReduceImplementation<false>::run(input, output_value, initial_value, reduction_operator, execution_policy_);
	}

	template<typename TView, typename TOperator>
	TOutputValue run(TView input, TOutputValue initial_value, TOperator reduction_operator) {
		TOutputValue val = initial_value;
		runAsync(input, val, initial_value, reduction_operator);
		return val;
	}

	ExecutionPolicy execution_policy_;

};

#ifdef __CUDACC__

template<typename TOutputValue>
class Reduce<TOutputValue, true> {
public:
	template<typename TSize>
	Reduce(TSize size, ExecutionPolicy execution_policy = ExecutionPolicy{}) :
		execution_policy_(execution_policy),
		tmp_vector_(detail::estimateTemporaryBuffer(size, execution_policy))
	{}

	template<typename TView, typename TOperator>
	void runAsync(TView input, TOutputValue &output_value, TOutputValue initial_value, TOperator reduction_operator) {
		detail::ReduceImplementation<true>::run(input, view(tmp_vector_), initial_value, reduction_operator, execution_policy_);
		view(tmp_vector_).getOnHost(0, output_value, execution_policy_.cuda_stream);
	}

	template<typename TView, typename TOperator>
	TOutputValue run(TView input, TOutputValue initial_value, TOperator reduction_operator) {
		TOutputValue val = initial_value;
		runAsync(input, val, initial_value, reduction_operator);
		BOLT_CHECK(cudaStreamSynchronize(execution_policy_.cuda_stream));
		return val;
	}

	ExecutionPolicy execution_policy_;
	DeviceImage<TOutputValue, 1> tmp_vector_;
};
#endif // __CUDACC__


template<typename TView, typename TOutputValue, typename TOperator>
TOutputValue reduce(TView inview, TOutputValue initial_value, TOperator reduction_operator, ExecutionPolicy execution_policy) {
	// TOutputValue output_value;
	// detail::ReduceImplementation<IsDeviceImageView<TView>::value>::run(inview, output_value, initial_value, reduction_operator, execution_policy);
	// BOLT_CHECK(cudaStreamSynchronize(execution_policy.cuda_stream));
	// return output_value;
	auto plan = Reduce<TOutputValue, IsDeviceImageView<TView>::value>(inview.size(), execution_policy);
	return plan.run(inview, initial_value, reduction_operator);
}

template<typename TView, typename TOutputView, typename TOutputValue, int tDimension, typename TOperator>
void dimensionReduce(TView inview, TOutputView output_view, DimensionValue<tDimension> dimension, TOutputValue initial_value, TOperator reduction_operator, ExecutionPolicy execution_policy) {
	detail::ReduceImplementation<IsDeviceImageView<TView>::value>::run(inview, output_view, dimension, initial_value, reduction_operator, execution_policy);
}

template<typename TView, typename TOutputValue, class>
TOutputValue sum(TView inview, TOutputValue initial_value, ExecutionPolicy execution_policy) {
	return reduce(inview, initial_value, thrust::plus<TOutputValue>(), execution_policy);
}

template<typename TView, class>
typename TView::Element sum(TView inview, ExecutionPolicy execution_policy) {
	// TODO(johny) - generic zero initialization
	return reduce(inview, typename TView::Element(0), thrust::plus<typename TView::Element>(), execution_policy);
}

template<typename TView1, typename TView2, typename TOutputValue>
TOutputValue sumSquareDifferences(TView1 view1, TView2 view2, TOutputValue initial_value, ExecutionPolicy execution_policy) {
	return sum(square(subtract(view1, view2)), initial_value, execution_policy);
}

/// Returns a minimum and maximum (in this order) values of the input view. Allows for
/// custom initialization of the starting min/max values.
template<typename TView>
Vector<typename TView::Element, 2> minMax(TView inview, Tuple<typename TView::Element, typename TView::Element> initial_value) {
	Tuple<typename TView::Element, typename TView::Element> min_max_tuple = reduce(zipViews(inview, inview), initial_value, MinMaxFunctor());
	return Vector<typename TView::Element, 2>(min_max_tuple.template get<0>(), min_max_tuple.template get<1>());
}

/// Returns a minimum and maximum (in this order) values of the input view.
template<typename TView>
Vector<typename TView::Element, 2> minMax(TView inview) {
	using MMPair = Tuple<typename TView::Element, typename TView::Element>;
	using Limits = ::std::numeric_limits<typename TView::Element>;
	return minMax(inview, MMPair(Limits::max(), Limits::min()));
}

/// Returns a minimum, maximum and mean (in this order) values of the input view. Allows for
/// custom initialization of the starting min/max/mean values.
template<typename TView, typename TMeanType>
Tuple<typename TView::Element, typename TView::Element, TMeanType> minMaxMean(TView inview, Tuple<typename TView::Element, typename TView::Element, TMeanType> initial_value) {
	Tuple<typename TView::Element, typename TView::Element, TMeanType> min_max_mean_tuple = reduce(zipViews(inview, inview, cast<TMeanType, TView>(inview)), initial_value, MinMaxMeanFunctor());
	min_max_mean_tuple.template get<2>() = min_max_mean_tuple.template get<2>() / (TMeanType)(inview.elementCount());
	return min_max_mean_tuple;
}

/// Returns a minimum, maximum and mean (in this order) values of the input view.
template<typename TView>
Tuple<typename TView::Element, typename TView::Element, double> minMaxMean(TView inview) {
	using MMMTriple = Tuple<typename TView::Element, typename TView::Element, double>;
	using Limits = ::std::numeric_limits<typename TView::Element>;
	return minMaxMean(inview, MMMTriple(Limits::max(), Limits::min(), 0.0));
}

struct CheckFloatIsFinite {
	BOLT_DECL_HYBRID bool operator()(bool intermediate, float value) const {
#ifdef __CUDA_ARCH__
		return intermediate && isfinite(value);
#else
		return intermediate && std::isfinite(value);
#endif //__CUDA_ARCH
	}

	BOLT_DECL_HYBRID bool operator()(bool intermediate1, bool intermediate2) const {
		return intermediate1 && intermediate2;
	}
};


template<typename TView, class>
bool isFinite(TView inview, ExecutionPolicy execution_policy) {
	return reduce(inview, true, CheckFloatIsFinite(), execution_policy);
}

}  // namespace bolt
