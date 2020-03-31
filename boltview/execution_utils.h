#pragma once


namespace bolt {

struct ExecutionPolicy {

#if defined(__CUDACC__)
	cudaStream_t cuda_stream = 0;
#else
	void * cuda_stream = nullptr;
#endif  // __CUDACC__
	int bucket_size = 4;
	int block_size = 512;
};

} // namespace bolt
