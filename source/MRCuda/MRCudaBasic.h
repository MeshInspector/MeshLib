#pragma once

#include "exports.h"

#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRExpected.h"

#include <thread>

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
MRCUDA_API size_t maxBufferSize( size_t availableBytes, size_t elementCount, size_t elementBytes );

// Returns maximum buffer size in elements that can be allocated with given memory limit
// The size is aligned to the block dimensions
MRCUDA_API size_t maxBufferSizeAlignedByBlock( size_t availableBytes, const Vector2i& blockDims, size_t elementBytes );
MRCUDA_API size_t maxBufferSizeAlignedByBlock( size_t availableBytes, const Vector3i& blockDims, size_t elementBytes );

// ...
template <typename BufferType, typename InputIt, typename GPUFunc, typename CPUFunc>
Expected<void> cudaPipeline( BufferType init, InputIt begin, InputIt end, GPUFunc gpuFunc, CPUFunc cpuFunc )
{
    std::array<BufferType, 2> buffers { init, init };
    std::array<InputIt, 2> it;
    enum Device
    {
        GPU = 0,
        CPU = 1,
    };

    for ( it[GPU] = begin; it[GPU] != end; it[CPU] = it[GPU]++ )
    {
        // TODO: replace with cudaStream usage
        Expected<void> gpuRes;
        auto gpuThread = std::jthread( [&]
        {
            gpuRes = gpuFunc( buffers[GPU], *it[GPU] );
        } );

        if ( it[GPU] != begin )
        {
            if ( auto cpuRes = cpuFunc( buffers[CPU], *it[CPU] ); !cpuRes )
                return cpuRes;
        }

        gpuThread.join();
        if ( !gpuRes )
            return gpuRes;

        std::swap( buffers[GPU], buffers[CPU] );
    }
    // process the last item
    return cpuFunc( buffers[CPU], *it[CPU] );
}

} //namespace Cuda

} //namespace MR
