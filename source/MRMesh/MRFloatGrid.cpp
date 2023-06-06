#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "MRFloatGrid.h"
#include "MRVector3.h"
#include "MRBitSet.h"
#include "MRVolumeIndexer.h"
#include "MRTimer.h"
#include "MRVDBConversions.h"

namespace MR
{

FloatGrid MakeFloatGrid( openvdb::FloatGrid::Ptr&& p )
{
    if ( !p )
        return {};
    return std::make_shared<OpenVdbFloatGrid>( std::move( *p ) );
}

FloatGrid resampled( const FloatGrid& grid, const Vector3f& voxelScale, ProgressCallback cb )
{
    if ( !grid )
        return {};
    const openvdb::FloatGrid & grid_ = *grid;
    MR_TIMER;
    openvdb::FloatGrid::Ptr dest = openvdb::FloatGrid::create();
    openvdb::Mat4R transform;
    transform.setToScale( openvdb::Vec3R{ voxelScale.x,voxelScale.y,voxelScale.z } );
    dest->setTransform( openvdb::math::Transform::createLinearTransform( transform ) ); // org voxel size is 1.0f

    ProgressCallback undefinedProgressCb;
    float i = 1.0f;
    if ( cb )
        undefinedProgressCb = [&] ( float )->bool
    {
        i += 1e-4f;
        return cb( 1.0f - 1.0f / std::sqrt( i ) );
    };

    ProgressInterrupter interrupter( undefinedProgressCb );
    // openvdb::util::NullInterrupter template argument to avoid tbb inconsistency
    openvdb::tools::resampleToMatch<openvdb::tools::BoxSampler, openvdb::util::NullInterrupter>( grid_, *dest, interrupter );
    if ( interrupter.getWasInterrupted() )
        return {};
    // restore normal scale
    dest->setTransform( openvdb::math::Transform::createLinearTransform( 1.0f ) );
    dest->setGridClass( grid->getGridClass() );

    return MakeFloatGrid( std::move( dest ) );
}

FloatGrid resampled( const FloatGrid& grid, float voxelScale, ProgressCallback cb )
{
    return resampled( grid, Vector3f::diagonal( voxelScale ), cb );
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
