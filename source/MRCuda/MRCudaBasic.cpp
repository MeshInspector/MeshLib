#include "MRCudaBasic.h"
#include "MRCudaBasic.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <MRPch/MRSpdlog.h>

namespace MR
{

namespace Cuda
{

bool isCudaAvailable()
{
    int n;
    cudaError err = cudaGetDeviceCount( &n );
    if ( err != cudaError::cudaSuccess )
        return false;
    return n > 0;
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
