#include "MRCudaBasic.h"
#include "MRCudaBasic.hpp"

#include "MRCuda.cuh"

#include <MRMesh/MRVector2.h>
#include <MRMesh/MRVector3.h>
#include <MRPch/MRSpdlog.h>

namespace MR
{

namespace Cuda
{

bool isCudaAvailable( int* driverVersionOut, int* runtimeVersionOut, int* computeMajorOut, int* computeMinorOut )
{
    int n;
    cudaError_t err = cudaGetDeviceCount( &n );
    if ( err != cudaSuccess )
        return false;
    if ( n <= 0 )
        return false;
    int driverVersion{ 0 };
    int runtimeVersion{ 0 };
    err = cudaDriverGetVersion( &driverVersion );
    if ( err != cudaSuccess )
        return false;
    
    err = cudaRuntimeGetVersion( &runtimeVersion );
    if ( err != cudaSuccess )
        return false;

    int computeMajor{ 0 };
    int computeMinor{ 0 };
    err = cudaDeviceGetAttribute( &computeMajor, cudaDevAttrComputeCapabilityMajor, 0 );
    if ( err != cudaSuccess )
        return false;
    err = cudaDeviceGetAttribute( &computeMinor, cudaDevAttrComputeCapabilityMinor, 0 );
    if ( err != cudaSuccess )
        return false;

    if ( driverVersionOut )
        *driverVersionOut = driverVersion;
    if ( runtimeVersionOut )
        *runtimeVersionOut = runtimeVersion;
    if ( computeMajorOut )
        *computeMajorOut = computeMajor;
    if ( computeMinorOut )
        *computeMinorOut = computeMinor;

    // according to https://en.wikipedia.org/wiki/CUDA Compute Capability (CUDA SDK support vs. Microarchitecture) table
    if ( runtimeVersion / 1000 >= 12 && computeMajor < 5 )
        return false;
    else if ( runtimeVersion / 1000 > 10 && ( computeMajor < 3 || ( computeMajor == 3 && computeMinor < 5 ) ) )
        return false;

    return runtimeVersion <= driverVersion;
}

size_t getCudaAvailableMemory()
{
    if ( !isCudaAvailable() )
        return 0;
    CUDA_EXEC( cudaSetDevice( 0 ) );
    size_t memFree = 0, memTot = 0;
    CUDA_EXEC( cudaMemGetInfo( &memFree, &memTot ) );
    // minus extra 128 MB
    return memFree - 128 * 1024 * 1024;
}

size_t getCudaSafeMemoryLimit()
{
    constexpr float cMaxGpuMemoryUsage = 0.80f;
    return size_t( (float)getCudaAvailableMemory() * cMaxGpuMemoryUsage );
}

size_t maxBufferSize( size_t availableBytes, size_t elementCount, size_t elementBytes )
{
    return std::min( availableBytes / elementBytes, elementCount );
}

size_t maxBufferSizeAlignedByBlock( size_t availableBytes, const Vector2i& blockDims, size_t elementBytes )
{
    const auto rowSize = (size_t)blockDims.x;
    return std::min( availableBytes / elementBytes / rowSize, (size_t)blockDims.y ) * rowSize;
}

size_t maxBufferSizeAlignedByBlock( size_t availableBytes, const Vector3i& blockDims, size_t elementBytes )
{
    const auto layerSize = (size_t)blockDims.x * blockDims.y;
    return std::min( availableBytes / elementBytes / layerSize, (size_t)blockDims.z ) * layerSize;
}

std::string getError( cudaError_t code )
{
    return fmt::format( "CUDA error: {}", cudaGetErrorString( code ) );
}

cudaError_t logError( cudaError_t code, const char * file, int line )
{
    if ( code == cudaSuccess )
        return code;

    if ( file )
    {
        spdlog::error("CUDA error {}: {}. In file: {} Line: {}", 
            cudaGetErrorName( code ), cudaGetErrorString( code ), file, line );
    }
    else
    {
        spdlog::error( "CUDA error {}: {}", cudaGetErrorName( code ), cudaGetErrorString( code ) );
    }
    return code;
}

} //namespace Cuda

} //namespace MR
