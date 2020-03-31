// Copyright 2018 Eyen SE
// Author: Pavel Mikuš pavel.mikus@eyen.se

#pragma once

#include <functional>
#include <future>
#include <iostream>  // std::cout
#include <memory>
#include <thread>  // std::thread
#include "cuda_utils.h"
#include <exception>
#include <stdexcept>


namespace bolt {


namespace detail {


template<typename ValueType>
void valueHolderSet(std::promise<ValueType>& promise, std::function<ValueType()> function) {
	promise.set_value(function());
}

template<>
void valueHolderSet<void>(std::promise<void>& promise, std::function<void()> function) {
	function();
	promise.set_value();
}


/// Struct to hold result of ansynchronous operation
template<typename ValueType>
class ValueHolder {
public:
	/// Inititalize the Value holder with resultSetter function that takes no arguments and returns result
	/// of ansynchronous operation
	ValueHolder(std::function<ValueType()> resultSetter) : resultSetter_(resultSetter) {
		future_ = std::shared_future<ValueType>(promise_.get_future());
	}

	void setResult() {
		try {
			valueHolderSet<ValueType>(promise_, resultSetter_);
		} catch (...) { // FIX(johny) - do not catch everything
			setException(std::current_exception());
		}
	}

	void setException(std::exception_ptr except_ptr) {
		promise_.set_exception(except_ptr);
	}

	std::shared_future<ValueType> getFuture() {
		return future_;
	}

	ValueType getValue() {
		return future_.get();
	}

	/// cuda callback to set result. The object must exist at the time of callback, and the result setter should be
	/// lightweight, since it blocks the stream
	static void CUDART_CB taskDoneCallback(cudaStream_t stream, cudaError_t status, void* data) {
		ValueHolder<ValueType>* valueHolder = static_cast<ValueHolder<ValueType>*>(data);
		try {
			BOLT_CHECK(status);
			BOLT_CHECK_ERROR_STATE("Problem in kernel before device future");
			valueHolder->setResult();
		} catch (...) {
			valueHolder->setException(std::current_exception());
		}
	}

	/// to ensure existence until the resultSetter (can be callback) is called
	~ValueHolder() {
		future_.wait();
	}

private:
	std::function<ValueType()> resultSetter_;
	std::promise<ValueType> promise_;
	std::shared_future<ValueType> future_;
};

}  // namespace detail

/// DeviceFuture provides mechanism to access result of ansynchronous operations
template<typename ValueType>
class DeviceFuture {
public:
	/// A callback is registered on the given stream, and the resultSetter is
	/// called after the stream reaches the callback
	DeviceFuture(cudaStream_t stream, std::function<ValueType()> resultSetter) {
		valueHolder_ = std::make_shared<detail::ValueHolder<ValueType>>(resultSetter);
		BOLT_CHECK(cudaStreamAddCallback(stream, detail::ValueHolder<ValueType>::taskDoneCallback, (void*)valueHolder_.get(), 0));
	}

	/// DeviceFuture can be followed by any number of tasks, all of which will be executed after the result of this DeviceFuture is available
	template<typename TCallable>
	auto Then(TCallable task) -> DeviceFuture<decltype(task(std::declval<ValueType>()))> {
		/// The mechanism of creating follow up tasks is to bind the shared pointer to the valueHolder as an argument of the next task.
		/// Using the shared pointer ensures existence of the ValueHolder even if the DeviceFuture instance is destroyed.
		/// Creates a new DeviceFuture that launches the resultSetter in new thread. This thread is blocked on the access to result of the
		/// previous DeviceFuture, until the result is available
		auto tempCopy = valueHolder_;
		auto resultSetter = [=]() { return task(tempCopy->getValue()); };
		using ReturnType = decltype(task(std::declval<ValueType>()));
		return DeviceFuture<ReturnType>(resultSetter);
	}

	void wait() {
		valueHolder_->getFuture().wait();
	}

	ValueType getValue() {
		return valueHolder_->getValue();
	}

	/// Creates DeviceFuture with resultSetter and immediately launches it in the new thread
	DeviceFuture(std::function<ValueType()> resultSetter) {
		valueHolder_ = std::make_shared<detail::ValueHolder<ValueType>>(resultSetter);
		std::thread([](std::shared_ptr<detail::ValueHolder<ValueType>> valueHolder) { valueHolder->setResult(); }, valueHolder_).detach();
	}

private:
	std::shared_ptr<detail::ValueHolder<ValueType>> valueHolder_;
};


/// void specialization of DeviceFuture
template<>
class DeviceFuture<void> {
public:
	DeviceFuture(cudaStream_t stream, std::function<void()> followup_work = []() {}) {
		valueHolder_ = std::make_shared<detail::ValueHolder<void>>(followup_work);
		BOLT_CHECK(cudaStreamAddCallback(stream, detail::ValueHolder<void>::taskDoneCallback, (void*)valueHolder_.get(), 0));
	}

	template<typename TCallable>
	auto Then(TCallable task) -> DeviceFuture<decltype(task())> {
		auto tempCopy = valueHolder_;
		auto resultSetter = [=]() {
			tempCopy->getValue();
			return task();
		};
		using ReturnType = decltype(task());
		return DeviceFuture<ReturnType>(resultSetter);
	}

	void wait() {
		valueHolder_->getFuture().wait();
	}

	void getValue() {
		valueHolder_->getValue();
	}

	DeviceFuture(std::function<void()> resultSetter) {
		valueHolder_ = std::make_shared<detail::ValueHolder<void>>(resultSetter);
		std::thread([](std::shared_ptr<detail::ValueHolder<void>> valueHolder) { valueHolder->setResult(); }, valueHolder_).detach();
	}

private:
	std::shared_ptr<detail::ValueHolder<void>> valueHolder_;
};

}  // namespace bolt
