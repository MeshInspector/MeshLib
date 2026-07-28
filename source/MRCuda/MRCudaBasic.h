#pragma once

#include "exports.h"

#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRExpected.h"

namespace MR
{

namespace Cuda
{

struct DeviceInfo
{
    /// maximum CUDA version supported by the driver
    int driverVersion = 0;

    /// application's CUDA version
    int runtimeVersion = 0;

    /// compute capability major version
    int computeMajor = 0;

    /// compute capability minor version
    int computeMinor = 0;

    /// global memory on device in bytes (not all is available for our process)
    size_t totalGlobalMem = 0;

    /// name of the device
    std::string name;

    /// returns true if all versions pass the checks
    [[nodiscard]] MRCUDA_API bool fitForComputations() const;
};

/// Returns an error if CUDA is not available
MRCUDA_API Expected<DeviceInfo> getDeviceInfo();

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
