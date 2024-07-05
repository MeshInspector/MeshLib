#pragma once
#include <cstddef>

#include "exports.h"

namespace MR
{

namespace Cuda
{

// Returns true if Cuda is present on this GPU
// optional out maximum driver supported version
// optional out current runtime version
MRCUDA_API bool isCudaAvailable( int* driverVersion = nullptr, int* runtimeVersion = nullptr );

// Returns available GPU memory in bytes
MRCUDA_API size_t getCudaAvailableMemory();

} //namespace Cuda

} //namespace MR
