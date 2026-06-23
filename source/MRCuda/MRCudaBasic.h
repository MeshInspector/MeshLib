#pragma once

#include "exports.h"

#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRExpected.h"

namespace MR
{

namespace Cuda
{

struct RuntimeInfo
{
    /// maximum driver supported version
    int driverVersion = 0;

    /// current runtime version
    int runtimeVersion = 0;

    /// compute capability major version
    int computeMajor = 0;

    /// compute capability minor version
    int computeMinor = 0;

    /// returns true if all versions pass the checks
    [[nodiscard]] MRCUDA_API bool fitForComputations() const;
};

/// Returns an error if CUDA is not available
MRCUDA_API Expected<RuntimeInfo> getRuntimeInfo();

/// Returns true if Cuda is present on this GPU
/// optional out maximum driver supported version
/// optional out current runtime version
/// optional out compute capability major version
/// optional out compute capability minor version
[[deprecated( "Use getRuntimeInfo")]] MRCUDA_API bool isCudaAvailable( int* driverVersion = nullptr, int* runtimeVersion = nullptr, int* computeMajor = nullptr, int* computeMinor = nullptr );

/// Returns available GPU memory in bytes
MRCUDA_API size_t getCudaAvailableMemory();

/// Returns maximum safe amount of free GPU memory that will be used for dynamic-sized buffers
MRCUDA_API size_t getCudaSafeMemoryLimit();

/// Returns maximum buffer size in elements that can be allocated with given memory limit
MRCUDA_API size_t maxBufferSize( size_t availableBytes, size_t elementCount, size_t elementBytes );

/// Returns maximum buffer size in elements that can be allocated with given memory limit
/// The size is aligned to the block dimensions
MRCUDA_API size_t maxBufferSizeAlignedByBlock( size_t availableBytes, const Vector2i& blockDims, size_t elementBytes );
MRCUDA_API size_t maxBufferSizeAlignedByBlock( size_t availableBytes, const Vector3i& blockDims, size_t elementBytes );

} //namespace Cuda

} //namespace MR
