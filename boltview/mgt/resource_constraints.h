// Copyright 2016 Eyen SE
// Author: Tomas Krupka tomas.krupka@eyen.se

#pragma once

#include <vector>

#include <boltview/device_properties.h>
#include <boltview/cuda_utils.h>

namespace bolt {

namespace mgt {

/// \addtogroup mgt Multi GPU Scheduling
/// @{

/// Struct used to define the size of tasks.
class ResourceConstraints {
public:
	/// throw a warning if the gpus are heterogeneous
	explicit ResourceConstraints(std::vector<bolt::DeviceProperties> device_properties) : device_properties_(device_properties) {}

	bolt::DeviceProperties getLeastCapableDeviceProperties() {
		auto result = std::min_element(device_properties_.begin(), device_properties_.end(),
			[](const bolt::DeviceProperties& first, bolt::DeviceProperties& second){
				return first.total_memory < second.total_memory;
			});
		return *result;
	}


private:
	/// a task should be generated to not bypass any of the device properties,
	/// at this point total memory and texture image size.
	std::vector<bolt::DeviceProperties> device_properties_;
};

/// @}

}  // namespace mgt

}  // namespace bolt
