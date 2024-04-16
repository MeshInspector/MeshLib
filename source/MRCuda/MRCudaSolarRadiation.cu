#include "MRCudaSolarRadiation.cuh"
#include "MRCudaBasic.h"
#include "device_launch_parameters.h"
#include "MRCudaFloat.cuh"
#include <float.h>

namespace MR
{
namespace Cuda
{

__global__ void rayKernel( const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces, const float3* samples, const uint64_t* validSamples, const IntersectionPrecomputes* precs, uint64_t* res, const size_t resBlockCount, const size_t sampleCount, const size_t precCount, MeshIntersectionResult* outIntersections )
{
    if ( resBlockCount == 0 )
    {
        assert( false );
        return;
    }

    const size_t blockIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if ( blockIndex >= resBlockCount )
        return;

    const size_t blockStart = ( blockIndex << 6 ); // use bit shift instead of multiplying on 64
    const size_t blockEnd = ( blockIndex == resBlockCount - 1) ? ( sampleCount * precCount ) : ( ( blockIndex + 1 ) << 6 );

    uint64_t currentBit = 1;
    uint64_t block = 0;
    for ( size_t index = blockStart; index < blockEnd; ++index )
    {
        const size_t sample = index / precCount;
        if ( testBit( validSamples, sample ) )
        {
            const size_t patch = index % precCount;
            const auto intersectRes = rayMeshIntersect( nodes, meshPoints, faces, samples[sample], 0, FLT_MAX, precs[patch], bool( outIntersections ) );
            if ( intersectRes.distanceAlongLine > 0 )
            {
                block |= currentBit;         
            }
            else if ( outIntersections )
            {
                outIntersections[index] = intersectRes;
            }
        }
        currentBit <<= 1;
    }

    res[blockIndex] = block;
}

__global__ void radiationKernel( const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces, const float3* samples, const uint64_t* validSamples, const SkyPatch* skyPatches, const IntersectionPrecomputes* precs, const float rMaxRadiation, float* res, const size_t sampleCount, const size_t precCount, MeshIntersectionResult* outIntersections )
{
    if ( sampleCount == 0 )
    {
        assert( false );
        return;
    }

    const size_t sampleVertId = blockIdx.x * blockDim.x + threadIdx.x;
    if ( sampleVertId >= sampleCount || !testBit( validSamples, sampleVertId ) )
        return;

    const auto samplePt = samples[sampleVertId];

    float totalRadiation = 0;
    for ( int i = 0; i < precCount; ++i )
    {
        const auto intersectRes = rayMeshIntersect( nodes, meshPoints, faces, samplePt, 0, FLT_MAX, precs[i], bool( outIntersections ) );
        if ( intersectRes.distanceAlongLine < 0 )
            totalRadiation += skyPatches[i].radiation;
        else if ( outIntersections )
            outIntersections[sampleVertId * precCount + i] = intersectRes;

    }

    res[sampleVertId] = rMaxRadiation * totalRadiation;
}

__global__ void radiationKernel( const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces, const float3* samples, const uint64_t* validSamples, const SkyPatch* skyPatches, const IntersectionPrecomputes* precs, const float rMaxRadiation, float* res, const size_t sampleCount, const size_t precCount, uint64_t* outSkyRays )
{
    if ( sampleCount == 0 )
    {
        assert( false );
        return;
    }

    const size_t sampleVertId = blockIdx.x * blockDim.x + threadIdx.x;
    if ( sampleVertId >= sampleCount || !testBit( validSamples, sampleVertId ) )
        return;
    
    float totalRadiation = 0;
    auto ray = size_t( sampleVertId ) * precCount;
    for ( int i = 0; i < precCount; ++i, ++ray )
    {
        if ( testBit( outSkyRays, ray ) )
            totalRadiation += skyPatches[i].radiation;
    }

    res[sampleVertId] = rMaxRadiation * totalRadiation;
}

cudaError_t findSkyRaysKernel( const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces, const float3* samples, const uint64_t* validSamples, const IntersectionPrecomputes* precs, uint64_t* res, const size_t resBlockCount, const size_t sampleCount, const size_t precCount, MeshIntersectionResult* outIntersections )
{
    constexpr int maxThreadsPerBlock = 640;
    int numBlocks = ( int( resBlockCount ) + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock;

    rayKernel << < numBlocks, maxThreadsPerBlock >> > ( nodes, meshPoints, faces, samples, validSamples, precs, res, resBlockCount, sampleCount, precCount, outIntersections );
    CUDA_EXEC_RETURN( cudaGetLastError() );

    return cudaSuccess;
}

cudaError_t computeSkyViewFactorKernel( const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces, const float3* samples, const uint64_t* validSamples, const SkyPatch* skyPatches, const IntersectionPrecomputes* precs, const float rMaxRadiation, float* res, const size_t sampleCount, const size_t precCount, MeshIntersectionResult* outIntersections )
{
    constexpr int maxThreadsPerBlock = 640;
    int numBlocks = (int( sampleCount ) + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
   
    radiationKernel << < numBlocks, maxThreadsPerBlock >> > (nodes, meshPoints, faces, samples, validSamples, skyPatches, precs, rMaxRadiation, res, sampleCount, precCount, outIntersections );

    CUDA_EXEC_RETURN( cudaGetLastError() );

    return cudaSuccess;
}

cudaError_t computeSkyViewFactorKernel( const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces, const float3* samples, const uint64_t* validSamples, const SkyPatch* skyPatches, const IntersectionPrecomputes* precs, const float rMaxRadiation, float* res, const size_t sampleCount, const size_t precCount, uint64_t* outSkyRays )
{
    constexpr int maxThreadsPerBlock = 640;
    int numBlocks = ( int( sampleCount ) + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock;
    radiationKernel << < numBlocks, maxThreadsPerBlock >> > ( nodes, meshPoints, faces, samples, validSamples, skyPatches, precs, rMaxRadiation, res, sampleCount, precCount, outSkyRays );
    CUDA_EXEC_RETURN( cudaGetLastError() );

    return cudaSuccess;
}

}
}