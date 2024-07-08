#include "MRCudaBasic.h"
#include "MRCudaBasic.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <MRPch/MRSpdlog.h>

namespace MR
{

namespace Cuda
{

bool isCudaAvailable( int* driverVersionOut, int* runtimeVersionOut, int* computeMajorOut, int* computeMinorOut )
{
    int n;
    cudaError err = cudaGetDeviceCount( &n );
    if ( err != cudaError::cudaSuccess )
        return false;
    if ( n <= 0 )
        return false;
    int driverVersion{ 0 };
    int runtimeVersion{ 0 };
    err = cudaDriverGetVersion( &driverVersion );
    if ( err != cudaError::cudaSuccess )
        return false;
    
    err = cudaRuntimeGetVersion( &runtimeVersion );
    if ( err != cudaError::cudaSuccess )
        return false;

    int computeMajor{ 0 };
    int computeMinor{ 0 };
    err = cudaDeviceGetAttribute( &computeMajor, cudaDevAttrComputeCapabilityMajor, 0 );
    if ( err != cudaError::cudaSuccess )
        return false;
    err = cudaDeviceGetAttribute( &computeMinor, cudaDevAttrComputeCapabilityMinor, 0 );
    if ( err != cudaError::cudaSuccess )
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
