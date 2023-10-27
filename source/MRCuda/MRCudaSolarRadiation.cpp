#include "MRCudaSolarRadiation.h"
#include "MRMesh/MRAABBTree.h"
#include "MRCudaSolarRadiation.cuh"

namespace MR
{
namespace Cuda
{

static void findMaxVectorDim( int& dimX, int& dimY, int& dimZ, const Vector3f& dir )
{
    if ( dir.x > dir.y )
    {
        if ( dir.x > dir.z )
        {
            if ( dir.y > dir.z )
            {
                // x>y>z
                if ( -dir.z > dir.x )
                {
                    dimZ = 2; dimX = 1; dimY = 0;
                }
                else
                {
                    dimZ = 0; dimX = 1; dimY = 2;
                }
            }
            else
            {
                // x>z>y
                if ( -dir.y > dir.x )
                {
                    dimZ = 1; dimX = 0; dimY = 2;
                }
                else
                {
                    dimZ = 0; dimX = 1; dimY = 2;
                }
            }
        }
        else
        {
            // z>x>y
            if ( -dir.y > dir.z )
            {
                dimZ = 1; dimX = 0; dimY = 2;
            }
            else
            {
                dimZ = 2; dimX = 0; dimY = 1;
            }
        }
    }
    else
    {
        if ( dir.y > dir.z )
        {
            if ( dir.x < dir.z )
            {
                // y>z>x
                if ( -dir.x > dir.y )
                {
                    dimZ = 0; dimX = 2; dimY = 1;
                }
                else
                {
                    dimZ = 1; dimX = 2; dimY = 0;
                }
            }
            else
            {
                // y>x>z
                if ( -dir.z > dir.y )
                {
                    dimZ = 2; dimX = 1; dimY = 0;
                }
                else
                {
                    dimZ = 1; dimX = 2; dimY = 0;
                }
            }
        }
        else
        {
            // z>y>x
            if ( -dir.x > dir.z )
            {
                dimZ = 0; dimX = 2; dimY = 1;
            }
            else
            {
                dimZ = 2; dimX = 0; dimY = 1;
            }
        }
    }
}


BitSet findSkyRays( const Mesh& terrain, const VertCoords& samples, const VertBitSet& validSamples, const std::vector<MR::SkyPatch>& skyPatches )
{
    const auto& tree = terrain.getAABBTree();
    const auto& nodes = tree.nodes();

    CUDA_EXEC( cudaSetDevice( 0 ) );
    //const size_t size = size_t( params.resolution.x ) * params.resolution.y;

    DynamicArray<Node3> cudaNodes;
    cudaNodes.fromVector( nodes.vec_ );

    DynamicArray<float3> cudaMeshPoints( terrain.points.vec_ );
    DynamicArray<FaceToThreeVerts> cudaFaces( terrain.topology.getTriangulation().vec_ );    
    DynamicArray<float3> cudaSamples( samples.vec_ );

    std::vector<uint64_t> blocks;
    boost::to_block_range( validSamples, std::back_inserter( blocks ) );
    DynamicArray<uint64_t> cudaValidSamples( blocks );

    std::vector<IntersectionPrecomputes> precs( skyPatches.size() );
    for ( size_t i = 0; i < skyPatches.size(); ++i )
    {
        const auto& dir = skyPatches[i].dir;
        auto& prec = precs[i];
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

    DynamicArray<IntersectionPrecomputes> cudaPrecs( precs );

    DynamicArray<uint64_t> cudaRes( samples.size() * skyPatches.size() );
    int trueBitCount = 0;
    findSkyRaysKernel( cudaNodes.data(), cudaMeshPoints.data(), cudaFaces.data(), cudaSamples.data(), cudaValidSamples.data(), cudaPrecs.data(), cudaRes.data(), cudaSamples.size(), cudaPrecs.size(), trueBitCount );

    std::vector<uint64_t> resBlocks;
    cudaRes.toVector( resBlocks );
    
    return { resBlocks.begin(), resBlocks.end() };
}

VertScalars  computeSkyViewFactor( const Mesh& terrain,
    const VertCoords& samples, const VertBitSet& validSamples,
    const std::vector<MR::SkyPatch>& skyPatches,
    BitSet* outSkyRays )
{
    const auto& tree = terrain.getAABBTree();
    const auto& nodes = tree.nodes();

    CUDA_EXEC( cudaSetDevice( 0 ) );
    //const size_t size = size_t( params.resolution.x ) * params.resolution.y;

    DynamicArray<Node3> cudaNodes;
    cudaNodes.fromVector( nodes.vec_ );

    DynamicArray<float3> cudaMeshPoints( terrain.points.vec_ );
    DynamicArray<FaceToThreeVerts> cudaFaces( terrain.topology.getTriangulation().vec_ );
    DynamicArray<float3> cudaSamples( samples.vec_ );

    std::vector<uint64_t> cudaValidSamplesblocks;
    boost::to_block_range( validSamples, std::back_inserter( cudaValidSamplesblocks ) );
    DynamicArray<uint64_t> cudaValidSamples( cudaValidSamplesblocks );

    std::vector<IntersectionPrecomputes> precs( skyPatches.size() );
    for ( size_t i = 0; i < skyPatches.size(); ++i )
    {
        const auto& dir = skyPatches[i].dir;
        auto& prec = precs[i];
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

    DynamicArray<IntersectionPrecomputes> cudaPrecs( precs );
    DynamicArray<SkyPatch> cudaSkyPatches( skyPatches );

    const size_t rayCount = samples.size() * skyPatches.size();
    DynamicArray<uint64_t> cudaOutSkyRays( rayCount / 64 + size_t( bool( rayCount % 64 ) ) );
    DynamicArray<float> cudaRes( samples.size() );

    float maxRadiation = 0;
    for ( const auto& patch : skyPatches )
        maxRadiation += patch.radiation;

    if ( outSkyRays )
    {
        int trueBitCount = 0;
        findSkyRaysKernel( cudaNodes.data(), cudaMeshPoints.data(), cudaFaces.data(), cudaSamples.data(), cudaValidSamples.data(), cudaPrecs.data(), cudaOutSkyRays.data(), cudaSamples.size(), cudaPrecs.size(), trueBitCount );
        const auto errCode = cudaGetLastError();
        if ( errCode != 0 )
            return {};

        std::vector<uint64_t> outSkyRaysBlocks;
        cudaOutSkyRays.toVector( outSkyRaysBlocks );
        *outSkyRays = BitSet( outSkyRaysBlocks.begin(), outSkyRaysBlocks.end() );
    }

    computeSkyViewFactorKernel(cudaNodes.data(), cudaMeshPoints.data(), cudaFaces.data(), cudaSamples.data(), cudaValidSamples.data(), cudaSkyPatches.data(), cudaPrecs.data(), 1.0f / maxRadiation, cudaRes.data(), cudaSamples.size(), cudaPrecs.size(), cudaOutSkyRays.data());    

    VertScalars res;
    cudaRes.toVector( res.vec_ );
    return res;

    /*std::vector<Node3> _cudaNodes;
    cudaNodes.toVector( _cudaNodes );

    std::vector<float3> _cudaMeshPoints;
    cudaMeshPoints.toVector( _cudaMeshPoints );

    std::vector<FaceToThreeVerts> _cudaFaces;
    cudaFaces.toVector( _cudaFaces );

    std::vector<float3> _cudaSamples;
    cudaSamples.toVector( _cudaSamples );

    std::vector<IntersectionPrecomputes> _cudaPrecs;
    cudaPrecs.toVector( _cudaPrecs );

    std::vector<SkyPatch> _cudaSkyPatches;
    cudaSkyPatches.toVector( _cudaSkyPatches );

    VertScalars res( _cudaSamples.size() );

    std::vector<uint64_t> outSkyRaysBlocks;
    if ( outSkyRays )
    {
        const size_t size = _cudaSamples.size() * _cudaSkyPatches.size();
        outSkyRaysBlocks.resize( size / 64 );

        for ( size_t index = 0; index < size; ++index )
        {
            const size_t sample = index / _cudaSkyPatches.size();
            if ( !testBit( cudaValidSamplesblocks.data(), sample) )
                continue;

            const size_t patch = index % _cudaSkyPatches.size();
            if ( index == 47125 )
            {
                index = index;
            }

            if ( rayMeshIntersect( _cudaNodes.data(), _cudaMeshPoints.data(), _cudaFaces.data(), _cudaSamples[sample], 0, FLT_MAX, precs[patch]) < 0 )
                setBit( outSkyRaysBlocks.data(), index);
        }

        for ( size_t index = 0; index < res.size(); ++index )
        {
            float totalRadiation = 0;
            auto ray = size_t( index ) * _cudaSkyPatches.size();
            for ( int i = 0; i < _cudaSkyPatches.size(); ++i, ++ray )
            {
                if ( testBit( outSkyRaysBlocks.data(), ray) )
                    totalRadiation += skyPatches[i].radiation;
            }

            res[VertId(index)] = totalRadiation / maxRadiation;
        }
    }*/

    //eturn res;
}

}
}