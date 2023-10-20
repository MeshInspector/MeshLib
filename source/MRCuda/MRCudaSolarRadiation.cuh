#include "MRCudaBasic.cuh"
#include "MRCudaMath.cuh"

namespace MR
{
namespace Cuda
{

struct IntersectionPrecomputes
{
    float3 invDir;
    int maxDimIdxZ = 2;
    int idxX = 0;
    int idxY = 1;
    int3 sign;
    float Sx, Sy, Sz;
};

void findSkyRaysKernel( const Node3* nodes, const float3* samples, const uint64_t* validSamples, const IntersectionPrecomputes* precs, uint64_t* res, const size_t nodeCount, const size_t sampleCount, const size_t precCount );

}
}