#include "MRCudaSolarRadiation.cuh"
#include "MRCudaBasic.h"
#include "device_launch_parameters.h"
#include "MRCudaFloat.cuh"
#include <float.h>

namespace MR
{
namespace Cuda
{



__global__ void rayKernel( const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces, const float3* samples, const uint64_t* validSamples, const IntersectionPrecomputes* precs, uint64_t* res, const size_t resBlockCount, const size_t sampleCount, const size_t precCount )
{
    if ( resBlockCount == 0 )
    {
        assert( false );
        return;
    }

    const size_t blockIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if ( blockIndex >= resBlockCount )
        return;

    const size_t blockStart = ( blockIndex << 6 );
    const size_t blockEnd = ( blockIndex == resBlockCount - 1) ? ( sampleCount * precCount ) : ( ( blockIndex + 1 ) << 6 );

    uint64_t currentBit = 1;
    for ( size_t index = blockStart; index < blockEnd; ++index )
    {
        const size_t sample = index / precCount;
        if ( testBit( validSamples, sample ) )
        {
            const size_t patch = index % precCount;

            if ( rayMeshIntersect( nodes, meshPoints, faces, samples[sample], 0, FLT_MAX, precs[patch] ) < 0 )
            {
                res[blockIndex] |= currentBit;         
            }
        }
        currentBit <<= 1;
    }
}

__global__ void radiationKernel( const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces, const float3* samples, const uint64_t* validSamples, const SkyPatch* skyPatches, const IntersectionPrecomputes* precs, const float rMaxRadiation, float* res, const size_t sampleCount, const size_t precCount, uint64_t* outSkyRays )
{
    if ( sampleCount == 0 )
    {
        assert( false );
        return;
    }

    const size_t sampleVertId = blockIdx.x * blockDim.x + threadIdx.x;
    if ( sampleVertId >= sampleCount )
        return;

    if ( outSkyRays )
    {
        float totalRadiation = 0;
        auto ray = size_t( sampleVertId ) * precCount;
        for ( int i = 0; i < precCount; ++i, ++ray )
        {
            if ( testBit( outSkyRays, ray ) )
                totalRadiation += skyPatches[i].radiation;
        }

        res[sampleVertId] = rMaxRadiation * totalRadiation;
        return;
    }

    const auto samplePt = samples[sampleVertId];

    float totalRadiation = 0;
    for ( int i = 0; i < precCount; ++i )
    {
        if ( rayMeshIntersect( nodes, meshPoints, faces, samplePt, 0, FLT_MAX, precs[i] ) < 0 )
            totalRadiation += skyPatches[i].radiation;
    }

    res[sampleVertId] = rMaxRadiation * totalRadiation;
}

void findSkyRaysKernel( const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces, const float3* samples, const uint64_t* validSamples, const IntersectionPrecomputes* precs, uint64_t* res, const size_t resBlockCount, const size_t sampleCount, const size_t precCount )
{
    constexpr int maxThreadsPerBlock = 640;
    int numBlocks = ( int( resBlockCount ) + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock;

    rayKernel << < numBlocks, maxThreadsPerBlock >> > ( nodes, meshPoints, faces, samples, validSamples, precs, res, resBlockCount, sampleCount, precCount );
}

void computeSkyViewFactorKernel( const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces, const float3* samples, const uint64_t* validSamples, const SkyPatch* skyPatches, const IntersectionPrecomputes* precs, const float rMaxRadiation, float* res, const size_t sampleCount, const size_t precCount, uint64_t* outSkyRays )
{
    constexpr int maxThreadsPerBlock = 640;
    int numBlocks = ( int( sampleCount ) + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock;
    radiationKernel << < numBlocks, maxThreadsPerBlock >> > ( nodes, meshPoints, faces, samples, validSamples, skyPatches, precs, rMaxRadiation, res, sampleCount, precCount, outSkyRays );
}

}
}