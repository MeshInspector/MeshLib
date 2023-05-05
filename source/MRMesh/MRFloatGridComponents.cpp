#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "MRFloatGridComponents.h"
#include "MRUnionFind.h"
#include "MRFloatGrid.h"
#include "MRVolumeIndexer.h"
#include "MRBitSet.h"
#include "MRTimer.h"

namespace MR
{

namespace FloatGridComponents
{

UnionFind<VoxelId> getUnionFindStructureVoxels( const FloatGrid& grid, const VolumeIndexer& indexer, const Vector3i& minVox, float isoValue )
{
    MR_TIMER;
    UnionFind<VoxelId> unionFindStructure( indexer.size() );
    auto accessor = grid->getConstAccessor();

    for ( int z = 0; z < indexer.dims().z; ++z )
        for ( int y = 0; y < indexer.dims().y; ++y )
            for ( int x = 0; x < indexer.dims().x; ++x )
            {
                auto thisVox = indexer.toVoxelId( { x,y,z } );
                auto thisCoord = minVox + Vector3i{ x, y, z };
                auto thisVal = accessor.getValue( { thisCoord.x, thisCoord.y, thisCoord.z } );
                for ( int i = 0; i < OutEdgeCount; i += 2 /*Plus dir only*/ )
                {
                    auto neighVox = indexer.getNeighbor( thisVox, OutEdge( i ) );
                    if ( !neighVox.valid() )
                        continue;
                    auto neighCoord = minVox + indexer.toPos( neighVox );

                    if ( ( thisVal < isoValue ) == ( accessor.getValue( { neighCoord.x, neighCoord.y, neighCoord.z } ) < isoValue ) )
                        unionFindStructure.unite( thisVox, neighVox );
                }
            }
    return unionFindStructure;
}

std::vector<VoxelBitSet> getAllComponents( const FloatGrid& grid, float isoValue /*= 0.0f*/ )
{
    MR_TIMER;
    auto bbox = grid->evalActiveVoxelBoundingBox();
    Vector3i dims = { bbox.dim().x(),bbox.dim().y(),bbox.dim().z() };
    VolumeIndexer indexer = VolumeIndexer( dims );
    auto unionFindStruct = getUnionFindStructureVoxels( grid, indexer, { bbox.min().x(),bbox.min().y(),bbox.min().z() }, isoValue );

    const auto& allRoots = unionFindStruct.roots();
    constexpr size_t InvalidRoot = ~size_t( 0 );
    std::vector<size_t> uniqueRootsMap( allRoots.size(), InvalidRoot );
    size_t k = 0;
    size_t curRoot;
    for ( size_t voxId = 0; voxId < indexer.size(); ++voxId )
    {
        curRoot = allRoots[VoxelId( voxId )];
        auto& uniqIndex = uniqueRootsMap[curRoot];
        if ( uniqIndex == InvalidRoot )
        {
            uniqIndex = k;
            ++k;
        }
    }
    std::vector<VoxelBitSet> res( k, VoxelBitSet( allRoots.size() ) );
    for ( size_t voxId = 0; voxId < indexer.size(); ++voxId )
    {
        curRoot = allRoots[VoxelId( voxId )];
        res[uniqueRootsMap[curRoot]].set( VoxelId( voxId ) );
    }
    return res;
}

}
}
#endif
