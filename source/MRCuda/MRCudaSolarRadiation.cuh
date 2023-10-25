#include "MRCudaBasic.cuh"
#include "MRCudaMath.cuh"

namespace MR
{
namespace Cuda
{

struct SkyPatch
{
    float3 dir;
    float radiation = 0;
};

void findSkyRaysKernel( const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces, const float3* samples, const uint64_t* validSamples, const IntersectionPrecomputes* precs, uint64_t* res, const size_t nodeCount, const size_t sampleCount, const size_t precCount );

void computeSkyViewFactorKernel( const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces, const float3* samples, const uint64_t* validSamples, const SkyPatch* skyPatches, const IntersectionPrecomputes* precs, const float rMaxRadiation, float* res, const size_t nodeCount, const size_t sampleCount, const size_t precCount, uint64_t* outSkyRays );
}
}