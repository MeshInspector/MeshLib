#pragma once

#include "exports.h"

#include "MRMesh/MRMeshFwd.h"

namespace MR
{

namespace Cuda
{

// Returns true if Cuda is present on this GPU
// optional out maximum driver supported version
// optional out current runtime version
// optional out compute capability major version
// optional out compute capability minor version
MRCUDA_API bool isCudaAvailable( int* driverVersion = nullptr, int* runtimeVersion = nullptr, int* computeMajor = nullptr, int* computeMinor = nullptr );

// Returns available GPU memory in bytes
MRCUDA_API size_t getCudaAvailableMemory();

// Returns maximum safe amount of free GPU memory that will be used for dynamic-sized buffers
MRCUDA_API size_t getCudaSafeMemoryLimit();

// Returns maximum buffer size in elements that can be allocated with given memory limit
MRCUDA_API size_t maxBufferSize( size_t availableBytes, size_t dim, size_t elementBytes );
MRCUDA_API size_t maxBufferSize( size_t availableBytes, const Vector2i& dims, size_t elementBytes );
MRCUDA_API size_t maxBufferSize( size_t availableBytes, const Vector3i& dims, size_t elementBytes );

} //namespace Cuda

} //namespace MR
