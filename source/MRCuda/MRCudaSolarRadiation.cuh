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

cudaError_t findSkyRaysKernel( const WideNode3* nodes, const float3* meshPoints, const LeafTri3* leafTris, const float3* samples, const uint64_t* validSamples, const IntersectionPrecomputes* precs, uint64_t* res, const size_t resBlockCount, const size_t sampleCount, const size_t precCount, MeshIntersectionResult* outIntersections );

cudaError_t computeSkyViewFactorKernel( const WideNode3* nodes, const float3* meshPoints, const LeafTri3* leafTris, const float3* samples, const uint64_t* validSamples, const SkyPatch* skyPatches, const IntersectionPrecomputes* precs, const float rMaxRadiation, float* res, const size_t sampleCount, const size_t precCount, MeshIntersectionResult* outIntersections );

cudaError_t computeSkyViewFactorKernel( const WideNode3* nodes, const float3* meshPoints, const LeafTri3* leafTris, const float3* samples, const uint64_t* validSamples, const SkyPatch* skyPatches, const IntersectionPrecomputes* precs, const float rMaxRadiation, float* res, const size_t sampleCount, const size_t precCount, uint64_t* outSkyRays );
}
}