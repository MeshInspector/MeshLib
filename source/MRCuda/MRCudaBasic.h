#pragma once
#include <cstddef>

#include "exports.h"

namespace MR
{

namespace Cuda
{

// Returns true if Cuda is present on this GPU
MRCUDA_API bool isCudaAvailable();

// Returns available GPU memory in bytes
MRCUDA_API size_t getCudaAvailableMemory();

} //namespace Cuda

} //namespace MR
