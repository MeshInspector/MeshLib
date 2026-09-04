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

namespace {

std::string cudaVersionString( int version )
{
    return fmt::format( "{}.{}", version / 1000, ( version % 1000 ) / 10 );
}

/// CUDA reports a missing driver and an outdated one alike, both as
/// cudaErrorInsufficientDriver whose message speaks only of an insufficient version
std::string driverDetails( cudaError_t code )
{
    if ( code != cudaErrorInsufficientDriver )
        return {};

    int driverVersion = 0, runtimeVersion = 0;
    cudaDriverGetVersion( &driverVersion ); // zero if no driver is installed
    cudaRuntimeGetVersion( &runtimeVersion );
    if ( driverVersion <= 0 )
        return fmt::format( " (no NVIDIA driver installed, CUDA runtime {})", cudaVersionString( runtimeVersion ) );
    return fmt::format( " (NVIDIA driver supports CUDA {}, older than the runtime {})",
        cudaVersionString( driverVersion ), cudaVersionString( runtimeVersion ) );
}

} // anonymous namespace

Expected<DeviceInfo> getDeviceInfo()
{
    DeviceInfo res;
    CUDA_RETURN_UNEXPECTED( cudaDriverGetVersion( &res.driverVersion ) );
    if ( res.driverVersion <= 0 )
        return MR::unexpected( "NVIDIA GPU error: no CUDA driver found" );

    {
        int n = 0;
        auto code = cudaGetDeviceCount( &n );
        if ( code != cudaSuccess || n <= 0 )
        {
            auto err = ( code != cudaSuccess ) ? MR::Cuda::getError( code ) : "NVIDIA GPU error: no capable device found";
            err += fmt::format( ", CUDA driver {}", cudaVersionString( res.driverVersion ) );
            return MR::unexpected( err );
        }
    }

    CUDA_RETURN_UNEXPECTED( cudaRuntimeGetVersion( &res.runtimeVersion ) );

    cudaDeviceProp prop;
    CUDA_RETURN_UNEXPECTED( cudaGetDeviceProperties( &prop, 0 ) );
    res.computeMajor = prop.major;
    res.computeMinor = prop.minor;
    res.totalGlobalMem = prop.totalGlobalMem;
    res.name = prop.name;

    return res;
}

bool DeviceInfo::fitForComputations() const
{
    // according to https://en.wikipedia.org/wiki/CUDA Compute Capability (CUDA SDK support vs. Microarchitecture) table
    if ( runtimeVersion / 1000 >= 12 && computeMajor < 5 )
        return false;
    if ( runtimeVersion / 1000 > 10 && ( computeMajor < 3 || ( computeMajor == 3 && computeMinor < 5 ) ) )
        return false;

    return runtimeVersion <= driverVersion;
}

bool isCudaAvailable( int* driverVersionOut, int* runtimeVersionOut, int* computeMajorOut, int* computeMinorOut )
{
     auto info = MR::Cuda::getDeviceInfo();
     if ( !info )
         return false;

    if ( driverVersionOut )
        *driverVersionOut = info->driverVersion;
    if ( runtimeVersionOut )
        *runtimeVersionOut = info->runtimeVersion;
    if ( computeMajorOut )
        *computeMajorOut = info->computeMajor;
    if ( computeMinorOut )
        *computeMinorOut = info->computeMinor;

    return info->fitForComputations();
}

size_t getCudaAvailableMemory()
{
    if ( CUDA_EXEC( cudaSetDevice( 0 ) ) != cudaSuccess )
        return 0;
    size_t memFree = 0, memTot = 0;
    if ( CUDA_EXEC( cudaMemGetInfo( &memFree, &memTot ) ) )
        return 0;
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
    return fmt::format( "NVIDIA GPU error: {}{}", cudaGetErrorString( code ), driverDetails( code ) );
}

cudaError_t logError( cudaError_t code, const char * file, int line )
{
    if ( code == cudaSuccess )
        return code;

    if ( file )
    {
        spdlog::error("CUDA error {}: {}{}. In file: {} Line: {}", 
            cudaGetErrorName( code ), cudaGetErrorString( code ), driverDetails( code ), file, line );
    }
    else
    {
        spdlog::error( "CUDA error {}: {}{}", cudaGetErrorName( code ), cudaGetErrorString( code ), driverDetails( code ) );
    }
    return code;
}

} //namespace Cuda

} //namespace MR
