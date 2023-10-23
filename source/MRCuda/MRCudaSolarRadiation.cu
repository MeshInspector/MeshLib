#include "MRCudaSolarRadiation.cuh"
#include "MRCudaBasic.h"
#include "device_launch_parameters.h"
#include "MRCudaFloat.cuh"
#include <float.h>

namespace MR
{
namespace Cuda
{

__global__ void kernel( const Node3* nodes, const float3* samples, const uint64_t* validSamples, const IntersectionPrecomputes* precs, uint64_t* res, const size_t nodeCount, const size_t sampleCount, const size_t precCount )
{
    if ( size == 0 )
    {
        assert( false );
        return;
    }

}

void findSkyRaysKernel( const Node3* nodes, const float3* samples, const uint64_t* validSamples, const IntersectionPrecomputes* precs, uint64_t* res, const size_t nodeCount, const size_t sampleCount, const size_t precCount )
{
    const size_t resSize = sampleCount * precCount;
    constexpr int maxThreadsPerBlock = 640;
    int numBlocks = ( int( resSize ) + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock;
}

}
}