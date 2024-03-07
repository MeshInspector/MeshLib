#include "MRCudaSolarRadiation.h"
#include "MRMesh/MRAABBTree.h"
#include "MRCudaSolarRadiation.cuh"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRMeshIntersect.h"
#include "MRMesh/MRIntersectionPrecomputes.h"

namespace MR
{
namespace Cuda
{

static std::vector<IntersectionPrecomputes> calcPrecs( const std::vector<MR::SkyPatch>& skyPatches )
{
    std::vector<IntersectionPrecomputes> precs;
    precs.reserve( skyPatches.size() );

    for ( const auto& skyPatch : skyPatches )
    {
        const auto& dir = skyPatch.dir;
        precs.emplace_back();
        auto& prec = precs.back();
        prec.dir = { .x = dir.x, .y = dir.y, .z = dir.z };

        findMaxVectorDim( prec.idxX, prec.idxY, prec.maxDimIdxZ, dir );

        prec.sign.x = dir.x >= 0.0f ? 1 : 0;
        prec.sign.y = dir.y >= 0.0f ? 1 : 0;
        prec.sign.z = dir.z >= 0.0f ? 1 : 0;

        prec.Sx = dir[prec.idxX] / dir[prec.maxDimIdxZ];
        prec.Sy = dir[prec.idxY] / dir[prec.maxDimIdxZ];
        prec.Sz = 1.0f / dir[prec.maxDimIdxZ];

        prec.invDir.x = ( dir.x == 0 ) ? std::numeric_limits<float>::max() : 1.0f / dir.x;
        prec.invDir.y = ( dir.y == 0 ) ? std::numeric_limits<float>::max() : 1.0f / dir.y;
        prec.invDir.z = ( dir.z == 0 ) ? std::numeric_limits<float>::max() : 1.0f / dir.z;
    }

    return precs;
}


BitSet findSkyRays( const Mesh& terrain, const VertCoords& samples, const VertBitSet& validSamples, const std::vector<MR::SkyPatch>& skyPatches, std::vector<MR::MeshIntersectionResult>* outIntersections )
{
    MR_TIMER

    const auto& tree = terrain.getAABBTree();
    const auto& nodes = tree.nodes();

    CUDA_EXEC( cudaSetDevice( 0 ) );

    DynamicArray<Node3> cudaNodes;
    cudaNodes.fromVector( nodes.vec_ );

    DynamicArray<float3> cudaMeshPoints( terrain.points.vec_ );
    DynamicArray<FaceToThreeVerts> cudaFaces( terrain.topology.getTriangulation().vec_ );    
    DynamicArray<float3> cudaSamples( samples.vec_ );

    std::vector<uint64_t> blocks;
    boost::to_block_range( validSamples, std::back_inserter( blocks ) );
    DynamicArray<uint64_t> cudaValidSamples( blocks );

    const auto precs = calcPrecs( skyPatches );
    DynamicArray<IntersectionPrecomputes> cudaPrecs( precs );

    const size_t rayCount = samples.size() * skyPatches.size();
    DynamicArray<uint64_t> cudaRes( ( rayCount + 63 ) / 64 );

    DynamicArray<MeshIntersectionResult> cudaIntersections;
    if ( outIntersections )
        cudaIntersections.resize( rayCount );

    findSkyRaysKernel( cudaNodes.data(), cudaMeshPoints.data(), cudaFaces.data(), cudaSamples.data(), cudaValidSamples.data(), cudaPrecs.data(), cudaRes.data(), cudaRes.size(), cudaSamples.size(), cudaPrecs.size(), cudaIntersections.data() );

    if ( outIntersections )
        cudaIntersections.toVector( *outIntersections );

    std::vector<uint64_t> resBlocks;
    cudaRes.toVector( resBlocks );    
    return { resBlocks.begin(), resBlocks.end() };
}

VertScalars  computeSkyViewFactor( const Mesh& terrain,
    const VertCoords& samples, const VertBitSet& validSamples,
    const std::vector<MR::SkyPatch>& skyPatches,
    BitSet* outSkyRays, std::vector<MR::MeshIntersectionResult>* outIntersections )
{
    MR_TIMER

    const auto& tree = terrain.getAABBTree();
    const auto& nodes = tree.nodes();

    CUDA_EXEC( cudaSetDevice( 0 ) );

    DynamicArray<Node3> cudaNodes;
    cudaNodes.fromVector( nodes.vec_ );

    DynamicArray<float3> cudaMeshPoints( terrain.points.vec_ );
    DynamicArray<FaceToThreeVerts> cudaFaces( terrain.topology.getTriangulation().vec_ );
    DynamicArray<float3> cudaSamples( samples.vec_ );

    std::vector<uint64_t> cudaValidSamplesblocks;
    boost::to_block_range( validSamples, std::back_inserter( cudaValidSamplesblocks ) );
    DynamicArray<uint64_t> cudaValidSamples( cudaValidSamplesblocks );

    const auto precs = calcPrecs( skyPatches );
    DynamicArray<IntersectionPrecomputes> cudaPrecs( precs );
    DynamicArray<SkyPatch> cudaSkyPatches( skyPatches );

    const size_t rayCount = samples.size() * skyPatches.size();
    DynamicArray<uint64_t> cudaOutSkyRays( ( rayCount + 63 ) / 64 );

    DynamicArray<MeshIntersectionResult> cudaIntersections;
    if ( outIntersections )
        cudaIntersections.resize( rayCount );

    if ( outSkyRays &&
        findSkyRaysKernel( cudaNodes.data(), cudaMeshPoints.data(), cudaFaces.data(), cudaSamples.data(), cudaValidSamples.data(), cudaPrecs.data(), cudaOutSkyRays.data(), cudaOutSkyRays.size(), cudaSamples.size(), cudaPrecs.size(), cudaIntersections.data() ) == cudaSuccess )
    {
        std::vector<uint64_t> outSkyRaysBlocks;
        cudaOutSkyRays.toVector( outSkyRaysBlocks );
        *outSkyRays = BitSet( outSkyRaysBlocks.begin(), outSkyRaysBlocks.end() );
    }

    /*std::vector<Node3> cudaNodesVec;
    cudaNodes.toVector( cudaNodesVec );

    std::vector<float3> cudaMeshPointsVec;
    cudaMeshPoints.toVector( cudaMeshPointsVec );

    std::vector<FaceToThreeVerts> cudaFacesVec;
    cudaFaces.toVector( cudaFacesVec );

    VertScalars res;
    res.resize( samples.size() );

    float maxRadiation = 0;
    for ( const auto& patch : skyPatches )
        maxRadiation += patch.radiation;

    if ( outIntersections )
    {
        outIntersections->resize( rayCount );
    }

    for ( VertId sampleVertId = VertId( 0 ); sampleVertId < samples.size(); ++sampleVertId )
    {
        if ( !validSamples[sampleVertId] )
            continue;

        const auto samplePt = samples[sampleVertId];
        float3 rayOrigin = { samplePt.x, samplePt.y, samplePt.z };

        float totalRadiation = 0;
        for ( int i = 0; i < skyPatches.size(); ++i )
        {
            const auto intersectRes = rayMeshIntersect( cudaNodesVec.data(), cudaMeshPointsVec.data(), cudaFacesVec.data(), rayOrigin, 0, FLT_MAX, precs[i]);
            if ( intersectRes.distanceAlongLine < 0 )
                totalRadiation += skyPatches[i].radiation;
            else
            {
                auto& mir = ( *outIntersections )[sampleVertId * skyPatches.size() + i];
                mir.distanceAlongLine = intersectRes.distanceAlongLine;
                mir.proj.face = FaceId( intersectRes.proj.faceId );
                
                mir.proj.point.x = intersectRes.proj.point.x;
                mir.proj.point.y = intersectRes.proj.point.y;
                mir.proj.point.z = intersectRes.proj.point.z;

                mir.mtp.bary.a = intersectRes.tp.a;
                mir.mtp.bary.b = intersectRes.tp.b;
            }
        }

        res[sampleVertId] = totalRadiation / maxRadiation;
    }

    return res;*/

    DynamicArray<float> cudaRes(samples.size());

    float maxRadiation = 0;
    for ( const auto& patch : skyPatches )
        maxRadiation += patch.radiation;

    if ( outSkyRays )
        computeSkyViewFactorKernel(cudaNodes.data(), cudaMeshPoints.data(), cudaFaces.data(), cudaSamples.data(), cudaValidSamples.data(), cudaSkyPatches.data(), cudaPrecs.data(), 1.0f / maxRadiation, cudaRes.data(), cudaSamples.size(), cudaPrecs.size(), cudaOutSkyRays.data() );
    else
        computeSkyViewFactorKernel( cudaNodes.data(), cudaMeshPoints.data(), cudaFaces.data(), cudaSamples.data(), cudaValidSamples.data(), cudaSkyPatches.data(), cudaPrecs.data(), 1.0f / maxRadiation, cudaRes.data(), cudaSamples.size(), cudaPrecs.size(), cudaIntersections.data() );
    
    if ( outIntersections )
        cudaIntersections.toVector( *outIntersections );

    VertScalars res;
    cudaRes.toVector( res.vec_ );
    return res;
}

}
}