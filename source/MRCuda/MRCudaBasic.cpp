#include "MRCudaBasic.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
    cudaSetDevice( 0 );
    size_t memFree = 0, memTot = 0;
    cudaMemGetInfo( &memFree, &memTot );
    // minus extra 128 MB
    return memFree - 128 * 1024 * 1024;
}

}

}