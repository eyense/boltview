// Copyright 2018 Eyen SE
// Author: Pavel Mikuš pavel.mikus@eyen.se

#pragma once

#include <boltview/cuda_utils.h>
#include <boltview/move_utils.h>

namespace bolt {
/// RAII wrapper for cudaEvent_t
class CudaEvent {
public:
	CudaEvent() {
		BOLT_CHECK(cudaEventCreate(&event_));
	}

	explicit CudaEvent(unsigned int flags) {
		BOLT_CHECK(cudaEventCreateWithFlags(&event_, flags));
	}

	CudaEvent(CudaEvent &&) = default;
	CudaEvent &operator=(CudaEvent &&) = default;

	/// Captures in the event the current content of the stream. Event moves into "occured" state when the captured tasksare complete
	/// Event and stream must be on the same device.
	void Record(cudaStream_t stream = 0) {
		BOLT_CHECK(cudaEventRecord(event_, stream));
	}

	/// Return whether the event has occured. If the event has NOT been recorded, returns true by default
	bool Query() {
		auto err = cudaEventQuery(event_);
		if (err == cudaSuccess) {
			return true;
		} else if (err == cudaErrorNotReady) {
			return false;
		} else {
			BOLT_CHECK_ERROR_STATE("Error querying the event");
			return false;
		}
	}

	/// Synchornizes the host with the event. If cudaEventBlockingSync flag was set, the thread is blocked,
	// otherwise, the thread will busy-wait (spin).
	void Synchronize() {
		BOLT_CHECK_MSG("The event has to be recorded", cudaEventSynchronize(event_));
	}

	/// Make a compute stream wait for this event
	void StreamWait(cudaStream_t stream) {
		// flag must be zero for now
		BOLT_CHECK(cudaStreamWaitEvent(stream, event_, 0));
	}

	/// Returns elapsed time between this event and the start event.
	float ElapsedTime(const CudaEvent &start) {
		float ms;
		BOLT_CHECK(cudaEventElapsedTime(&ms, start.get(), event_));
		return ms;
	}

	cudaEvent_t get() const {
		return event_;
	}

	CudaEvent(const CudaEvent &) = delete;
	CudaEvent &operator=(const CudaEvent &) = delete;

	~CudaEvent() {
		if (flag_.IsValid()) {
			try {
				BOLT_CHECK(cudaEventDestroy(event_));
			} catch (CudaError &e) {
				BOLT_ERROR_FORMAT("cudaEvent destruction failure.: %1%", boost::diagnostic_information(e));
			}
		}
	}

private:
	cudaEvent_t event_;
	MovedFromFlag flag_;
};

}  // namespace bolt
