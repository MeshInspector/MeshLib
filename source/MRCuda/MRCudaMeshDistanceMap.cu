#include "MRCudaMeshDistanceMap.cuh"

namespace MR
{

namespace Cuda
{

__global__ void kernel( const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces, MeshToDistanceMapParams params,
                        IntersectionPrecomputes prec, float shift, float* res, MeshTriPoint* outSamples, unsigned size, float3 xStep, float3 yStep )
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( index >= size )
        return;

    int x = index % params.resolution.x;
    int y = index / params.resolution.x;

    float3 org = params.orgPoint + xStep * ( x + 0.5f ) + yStep * ( y + 0.5f );

    MeshIntersectionResult interRes = rayMeshIntersect( nodes, meshPoints, faces, org, -FLT_MAX, FLT_MAX, prec );

    res[index] = -FLT_MAX;
    if ( !params.useDistanceLimits
         || ( interRes.distanceAlongLine < params.minValue )
         || ( interRes.distanceAlongLine > params.maxValue ) )
    {
        res[index] = interRes.distanceAlongLine - shift;
        if ( outSamples )
            outSamples[index] = interRes.tp;
    }
}


cudaError_t computeMeshDistanceMapKernel( 
    const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces, MeshToDistanceMapParams params,
    IntersectionPrecomputes prec, float shift,
    float* res, MeshTriPoint* outSamples )
{
    unsigned size = unsigned(params.resolution.x) * params.resolution.y;
    constexpr size_t maxThreadsPerBlock = 640;
    unsigned numBlocks = ( size + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock;

    float3 xStep = params.xRange / float( params.resolution.x );
    float3 yStep = params.yRange / float( params.resolution.y );

    // kernel
    kernel << < numBlocks, maxThreadsPerBlock >> > ( nodes, meshPoints, faces, params, prec, shift, res, outSamples, size, xStep, yStep );

    return cudaGetLastError();
}

}

}