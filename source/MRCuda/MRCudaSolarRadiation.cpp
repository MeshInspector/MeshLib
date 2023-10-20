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


BitSet findSkyRays( const Mesh& terrain, const VertCoords& samples, const VertBitSet& validSamples, const std::vector<SkyPatch>& skyPatches )
{
    const auto& tree = terrain.getAABBTree();
    const auto& nodes = tree.nodes();

    CUDA_EXEC( cudaSetDevice( 0 ) );
    //const size_t size = size_t( params.resolution.x ) * params.resolution.y;

    DynamicArray<Node3> cudaNodes;
    cudaNodes.fromVector( nodes.vec_ );

    DynamicArray<float3> cudaSamples;
    cudaSamples.fromVector( samples.vec_ );

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
    findSkyRaysKernel( cudaNodes.data(), cudaSamples.data(), cudaValidSamples.data(), cudaPrecs.data(), cudaRes.data(), cudaNodes.size(), cudaSamples.size(), cudaPrecs.size() );

    std::vector<uint64_t> resBlocks;
    cudaRes.toVector( resBlocks );
    
    return { resBlocks.begin(), resBlocks.end() };
}

}
}