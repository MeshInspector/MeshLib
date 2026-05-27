#include <MRMesh/MRVolumeIndexer.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRVector3.h>
#include "MRGTest.h"

namespace MR
{

TEST( MRMesh, ExpandShrinkVoxels )
{
    VolumeIndexer indexer( Vector3i::diagonal( 8 ) );
    VoxelBitSet mask( indexer.size() );
    mask.set( indexer.toVoxelId( { 4, 4, 4 } ) );
    mask.set( indexer.toVoxelId( { 4, 4, 5 } ) );

    VoxelBitSet refMask = mask;
    refMask.set( indexer.toVoxelId( { 4, 4, 3 } ) );
    refMask.set( indexer.toVoxelId( { 4, 5, 4 } ) );
    refMask.set( indexer.toVoxelId( { 5, 4, 4 } ) );
    refMask.set( indexer.toVoxelId( { 4, 3, 4 } ) );
    refMask.set( indexer.toVoxelId( { 3, 4, 4 } ) );

    refMask.set( indexer.toVoxelId( { 4, 4, 6 } ) );
    refMask.set( indexer.toVoxelId( { 4, 5, 5 } ) );
    refMask.set( indexer.toVoxelId( { 5, 4, 5 } ) );
    refMask.set( indexer.toVoxelId( { 4, 3, 5 } ) );
    refMask.set( indexer.toVoxelId( { 3, 4, 5 } ) );

    auto storeMask = mask;
    expandVoxelsMask( mask, indexer );
    EXPECT_TRUE( mask.is_subset_of( refMask ) );
    shrinkVoxelsMask( mask, indexer );
    EXPECT_TRUE( mask.is_subset_of( storeMask ) );
}

} //namespace MR
