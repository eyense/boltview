#pragma once

namespace bolt {

Reduce();

template<typename TView, typename TOutputView, typename TOutputValue, int tDimension, typename TOperator>
void DimensionReduce(
		TView view,
		TOutputView output_view,
		DimensionValue<tDimension> dimension,
		TOutputValue initial_value,
		TOperator reduction_operator,
		ExecutionPolicy execution_policy = ExecutionPolicy{})
{
	IsImageStack<TView>::value;


}

template<bool tRunOnDevice>
struct ReduceImplementation {
#ifdef __CUDACC__
	template<typename TView, typename TOutputValue, typename TOperator>
	static TOutputValue Run(TView view, TOutputValue initial_value, TOperator reduction_operator, ExecutionPolicy execution_policy) {
		constexpr int kBucketSize = 4;  // Bundle more computation in one block - prevents thread idling. TODO(johny) - should be specified by call policy.
		dim3 block(512, 1, 1);
		dim3 grid(1 + (view.elementCount() - 1) / (block.x * kBucketSize), 1, 1);

		thrust::device_vector<TOutputValue> tmp_vector(grid.x);

		ReduceKernel<TView, TOperator, TOutputValue, 512><<<grid, block, 0, execution_policy.cuda_stream>>>(view, initial_value, reduction_operator, tmp_vector.data().get());
		BOLT_CHECK_ERROR_AFTER_KERNEL("ReduceKernel", grid, block);
		BOLT_CHECK(cudaStreamSynchronize(execution_policy.cuda_stream));
		return thrust::reduce(thrust::cuda::par.on(execution_policy.cuda_stream), tmp_vector.begin(), tmp_vector.end(), initial_value, reduction_operator);
	}

	template<typename TView, typename TOutputView, typename TOutputValue, int tDimension, typename TOperator>
	static void Run(TView view, TOutputView output_view, DimensionValue<tDimension> dimension, TOutputValue initial_value, TOperator reduction_operator, ExecutionPolicy execution_policy) {
		auto size = view.size();
		auto reduced_size = RemoveDimension(size, tDimension);

		if (reduced_size != output_view.size()) {
			BOLT_THROW(IncompatibleViewSizes() << getViewPairSizesErrorInfo(reduced_size, output_view.size()));
		}

		constexpr int kBlockSize = 256;
		constexpr int kBucketSize = 2;
		int block_count = 1 + (size[tDimension] - 1) / (kBlockSize * kBucketSize);
		// TODO - better setup of bucket size and block size depending on input size
		if (block_count > 1) {
			dim3 block(kBlockSize, 1, 1);
			dim3 grid(block_count, reduced_size[0], reduced_size.kDimension >= 2 ? reduced_size[1] : 1);

			auto tmp_size = size;
			tmp_size[tDimension] = block_count;
			DeviceImage<TOutputValue, TView::kDimension> tmp_buffer(tmp_size);
			DimensionReduceKernel<TView, decltype(tmp_buffer.view()), TOutputValue, TOperator, tDimension, kBlockSize><<<grid, block, 0, execution_policy.cuda_stream>>>(view, tmp_buffer.view(), initial_value, reduction_operator);
			BOLT_CHECK_ERROR_AFTER_KERNEL("DimensionReduceKernel tmp buffer", grid, block);
			Run(tmp_buffer.constView(), output_view, dimension, initial_value, reduction_operator, execution_policy);
		} else {
			dim3 block(kBlockSize, 1, 1);
			dim3 grid(1 + (size[tDimension] - 1) / block.x, reduced_size[0], reduced_size.kDimension >= 2 ? reduced_size[1] : 1);

			DimensionReduceKernel<TView, TOutputView, TOutputValue, TOperator, tDimension, kBlockSize><<<grid, block, 0, execution_policy.cuda_stream>>>(view, output_view, initial_value, reduction_operator);
			BOLT_CHECK_ERROR_AFTER_KERNEL("DimensionReduceKernel", grid, block);
			BOLT_CHECK(cudaStreamSynchronize(execution_policy.cuda_stream));
		}
	}
#endif // __CUDACC__
};



}  // namespace bolt
