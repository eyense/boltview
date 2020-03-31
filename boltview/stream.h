// Copyright 2016 Eyen SE
// Author: Pavel Mikuš pavel.mikus@eyen.se
// Author: Tomas Krupka tomas.krupka@eyen.se

#pragma once

#include <boltview/cuda_utils.h>
#include <boltview/move_utils.h>


namespace bolt {

/// RAII wrapper for cudaStream_t
class CudaStream {
	public:
	CudaStream() {
		BOLT_CHECK(cudaStreamCreate(&stream_));
	}

	explicit CudaStream(unsigned int flags) {
		BOLT_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
	}

	CudaStream(unsigned int flags, int priority) {
		BOLT_CHECK(cudaStreamCreateWithPriority(&stream_, flags, priority));
	}

	CudaStream(CudaStream &&) = default;
	CudaStream &operator=(CudaStream &&) = default;

	/// Query the stream for completion status. True if all the work on the stream is completed
	bool query() {
		auto err = cudaStreamQuery(stream_);
		if (err == cudaSuccess) {
			return true;
		} else if (err == cudaErrorNotReady) {
			return false;
		} else {
			BOLT_CHECK_ERROR_STATE("Error querying the stream for completion status");
			return false;
		}
	}

	/// waits for stream tasks to complete
	void synchronize() {
		BOLT_CHECK(cudaStreamSynchronize(stream_));
	}

	/// makes the stream wait for the event
	void waitForEvent(cudaEvent_t event) {
		// flag must be zero for now
		BOLT_CHECK(cudaStreamWaitEvent(stream_, event, 0));
	}

	/// Adds callback after the current items. The callback will block further work until finished,
	/// and must NOT make any CUDA API calls.
	void addCallback(cudaStreamCallback_t callback, void *userData) {
		// flag must be zero for now
		BOLT_CHECK(cudaStreamAddCallback(stream_, callback, userData, 0));
	}

	cudaStream_t get() {
		return stream_;
	}

	CudaStream(const CudaStream &) = delete;
	CudaStream &operator=(const CudaStream &) = delete;

	~CudaStream() {
		if (flag_.isValid()) {
			try {
				BOLT_CHECK(cudaStreamDestroy(stream_));
			} catch (CudaError &e) {
				BOLT_ERROR_FORMAT("cudaStream destruction failure.: %1%", boost::diagnostic_information(e));
			}
		}
	}

	private:
	cudaStream_t stream_;
	MovedFromFlag flag_;
};

}  // namespace bolt
