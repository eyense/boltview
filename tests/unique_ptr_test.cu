// Copyright 2017 Eyen SE
// Author: Lukas Marsalek lukas.marsalek@eyen.eu

#define BOOST_TEST_MODULE UniquePtrTest
#include <boost/test/included/unit_test.hpp>
#include <tests/test_utils.h>
#include <cuda.h>
#include <boltview/unique_ptr.h>

namespace bolt {

template<class TType>
BOLT_GLOBAL
void FillValue(TType* ptr, TType value){
	*ptr = value;
}

template<class TType>
TType RunMemFillKernel(TType* devicePtr, TType testValue){
	FillValue<<<1, 64>>>(devicePtr, testValue);
	BOLT_CHECK_ERROR_STATE("FillValue");
	BOLT_CHECK(cudaDeviceSynchronize());
	int* gpuValuePtr;
	BOLT_CHECK(cudaHostAlloc(&gpuValuePtr, sizeof(int), cudaHostAllocDefault));
	BOLT_CHECK(cudaMemcpy(gpuValuePtr, devicePtr, sizeof(int), cudaMemcpyDeviceToHost));
	BOLT_CHECK(cudaDeviceSynchronize());
	int returnValue = *gpuValuePtr;
	BOLT_CHECK(cudaFreeHost(gpuValuePtr));
	return returnValue;
}

template<class TType>
BOLT_GLOBAL
void DereferenceOnDevice(TType* outArray, TType allocFailValue){
	int* devPtr = reinterpret_cast<int*>(malloc(sizeof(int)));
	if(devPtr == NULL){
		outArray[threadIdx.x] = allocFailValue;
	}
	*devPtr = threadIdx.x;
	device::unique_ptr<int> ptr(devPtr);
	outArray[threadIdx.x] = *ptr;
}

BOLT_AUTO_TEST_CASE(Empty) {
	device::unique_ptr<int> emptyPtr;
	BOOST_CHECK(emptyPtr.get() == nullptr);
}

BOLT_AUTO_TEST_CASE(ConstructionAndGet){
	int* devPtr = nullptr;
	BOLT_CHECK(cudaMalloc(&devPtr, sizeof(int)));
	device::unique_ptr<int> ptr(devPtr);
	int testValue = 3;
	int valueFromGPU = RunMemFillKernel(ptr.get(), testValue);
	BOOST_CHECK_EQUAL(valueFromGPU, testValue);
	BOOST_CHECK_EQUAL(ptr.get(), devPtr);
}

BOLT_AUTO_TEST_CASE(DeleteOnDestruction){
	int* devPtr = nullptr;
	BOLT_CHECK(cudaMalloc(&devPtr, sizeof(int)));
	{
		device::unique_ptr<int> ptr(devPtr);
	}
	bool wasDeleted = cudaFree(devPtr) != cudaSuccess;
	cudaGetLastError();
	BOOST_CHECK(wasDeleted);
}

BOLT_AUTO_TEST_CASE(Release){
	int* devPtr = nullptr;
	BOLT_CHECK(cudaMalloc(&devPtr, sizeof(int)));
	device::unique_ptr<int> ptr(devPtr);
	bool releasePtrUnmodified = ptr.release() == devPtr;
	BOOST_CHECK(releasePtrUnmodified);
	bool canBeDeleted = cudaFree(devPtr) == cudaSuccess;
	cudaGetLastError();
	BOOST_CHECK(canBeDeleted);
	bool nullAfterRelease = ptr.get() == nullptr;
	BOOST_CHECK(nullAfterRelease);
}

BOLT_AUTO_TEST_CASE(Reset){
	int* devPtr = nullptr;
	BOLT_CHECK(cudaMalloc(&devPtr, sizeof(int)));
	device::unique_ptr<int> ptr(devPtr);
	ptr.reset();
	bool nullAfterReset = ptr.get() == nullptr;
	BOOST_CHECK(nullAfterReset);
	bool deletedAfterReset = cudaFree(devPtr) != cudaSuccess;
	cudaGetLastError();
	BOOST_CHECK(deletedAfterReset);
}

BOLT_AUTO_TEST_CASE(BoolOperator){
	int* devPtr;
	BOLT_CHECK(cudaMalloc(&devPtr, sizeof(int)));
	device::unique_ptr<int> ptr(devPtr);
	if(!ptr){
		BOOST_CHECK(false);
	}
	ptr.reset();
	if(ptr){
		BOOST_CHECK(false);
	}
}

BOLT_AUTO_TEST_CASE(MoveOperator){
	auto devPtrA = device::make_unique<int>();
	auto devPtrB = device::make_unique<int>();
	int* intPtrB = devPtrB.get();
	devPtrA = std::move(devPtrB);
	BOOST_CHECK(devPtrA.get() == intPtrB);
	BOOST_CHECK(devPtrB.get() == nullptr);
}

BOLT_AUTO_TEST_CASE(Swap){
	int* devPtrA;
	BOLT_CHECK(cudaMalloc(&devPtrA, sizeof(int)));
	device::unique_ptr<int> ptrA(devPtrA);
	int* devPtrB;
	BOLT_CHECK(cudaMalloc(&devPtrB, sizeof(int)));
	device::unique_ptr<int> ptrB(devPtrB);
	ptrA.swap(ptrB);
	BOOST_CHECK(ptrA.get() == devPtrB);
	BOOST_CHECK(ptrB.get() == devPtrA);
}

BOLT_AUTO_TEST_CASE(DeviceUsage){
	int* devArray = nullptr;
	int blockSize = 64;
	BOLT_CHECK(cudaMalloc(&devArray, blockSize * sizeof(int)));
	int allocFailValue = 10;
	DereferenceOnDevice<<<1, blockSize>>>(devArray, allocFailValue);
	BOLT_CHECK_ERROR_STATE("DereferenceOnDevice");
	BOLT_CHECK(cudaDeviceSynchronize());
	int* hostArray = nullptr;
	BOLT_CHECK(cudaHostAlloc(&hostArray, blockSize * sizeof(int), cudaHostAllocDefault));
	BOLT_CHECK(cudaMemcpy(hostArray, devArray, blockSize * sizeof(int), cudaMemcpyDeviceToHost));
	for(int i = 0; i < blockSize; ++i){
			BOOST_CHECK_EQUAL(hostArray[i], i);
	}
	BOLT_CHECK(cudaFreeHost(hostArray));
}

BOLT_AUTO_TEST_CASE(MakeUnique){
	auto ptr = device::make_unique<int>();
	int testValue = 3;
	int valueFromGPU = RunMemFillKernel(ptr.get(), testValue);
	BOOST_CHECK_EQUAL(valueFromGPU, testValue);
}

}  // namespace bolt
