#ifndef MRMESH_NO_VOXEL
#include "MRFloatGrid.h"
#include "MRVector3.h"
#include "MRBitSet.h"
#include "MRVolumeIndexer.h"
#include "MRTimer.h"

namespace MR
{

FloatGrid MakeFloatGrid( openvdb::FloatGrid::Ptr&& p )
{
    if ( !p )
        return {};
    return std::make_shared<OpenVdbFloatGrid>( std::move( *p ) );
}

FloatGrid resampled( const FloatGrid & grid, const Vector3f& voxelScale )
{
    if ( !grid )
        return {};
    const openvdb::FloatGrid & grid_ = *grid;
    MR_TIMER;
    openvdb::FloatGrid::Ptr dest = openvdb::FloatGrid::create();
    openvdb::Mat4R transform;
    transform.setToScale( openvdb::Vec3R{ voxelScale.x,voxelScale.y,voxelScale.z } );
    dest->setTransform( openvdb::math::Transform::createLinearTransform( transform ) ); // org voxel size is 1.0f
    openvdb::tools::resampleToMatch<openvdb::tools::BoxSampler>( grid_, *dest );
    // restore normal scale
    dest->setTransform( openvdb::math::Transform::createLinearTransform( 1.0f ) );

    return MakeFloatGrid( std::move( dest ) );
}

FloatGrid resampled( const FloatGrid& grid, float voxelScale )
{
    return resampled( grid, Vector3f::diagonal( voxelScale ) );
}

void setValue( FloatGrid & grid, const VoxelBitSet& region, float value )
{
    if ( !grid )
        return;
    MR_TIMER;
    auto bbox = grid->evalActiveVoxelBoundingBox();
    Vector3i dims = { bbox.dim().x(),bbox.dim().y(),bbox.dim().z() };
    VolumeIndexer indexer = VolumeIndexer( dims );
    auto minVox = bbox.min();
    auto accessor = grid->getAccessor();
    for ( auto voxid : region )
    {
        auto pos = indexer.toPos( voxid );
        auto coord = minVox + openvdb::Coord{ pos.x,pos.y,pos.z };
        accessor.setValue( coord, value );
    }
}

}
#endif
