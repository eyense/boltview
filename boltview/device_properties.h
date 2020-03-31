#pragma once

#include <boltview/math/vector.h>

namespace bolt {


/// \addtogroup Utilities
/// @{

struct DeviceProperties {
	uint64_t total_memory = 0;
	Int3 max_texture; /// max texture size (on Titan 4096x4096x4096)
	Int3 max_texture_alt; /// alternative max texture size (on Titan 2048x2048x16384)
	int texture_pitch_alignment = 0;
};

struct DeviceMemoryInfo {
	size_t free_memory = 0;
	size_t total_memory = 0;
	int device = -1;
};

}  // namespace bolt

