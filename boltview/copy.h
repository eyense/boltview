// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#pragma once

#include <boltview/cuda_utils.h>
#include <boltview/exceptions.h>
#include <boltview/image_view_utils.h>

namespace bolt {

/// Tag for distiction between copy directions
template<bool tIsDevice, bool tToDevice>
struct CopyDirectionTag {};

using DeviceToDeviceTag = CopyDirectionTag<true, true>;
using DeviceToHostTag = CopyDirectionTag<true, false>;
using HostToHostTag = CopyDirectionTag<false, false>;
using HostToDeviceTag = CopyDirectionTag<false, true>;


template <typename TFromView, typename TToView>
struct IsImageViewPair {
	// NOLINTNEXTLINE(readability-identifier-naming)
	static constexpr bool value = IsImageView<TFromView>::value && IsImageView<TToView>::value;
};

/// Asynchronous copy between compatible image views.
/// Device/host direction is defined by the type of these views.
/// Copying views between host <-> device must be done through memory based views.
/// \param from_view Source
/// \param to_view Target
/// \param cuda_stream Selected CUDA stream.
template <typename TFromView, typename TToView/*, class = typename std::enable_if<IsImageViewPair<TFromView, TToView>::value>::type*/>
void copyAsync(
	TFromView from_view,
	TToView to_view,
	cudaStream_t cuda_stream = nullptr);


/// Synchronous copy between compatible image views.
/// \sa CopyAsync()
template <typename TFromView, typename TToView>
void copy(
	TFromView from_view,
	TToView to_view,
	cudaStream_t cuda_stream = nullptr);

}  // namespace bolt

#include "copy.tcc"
