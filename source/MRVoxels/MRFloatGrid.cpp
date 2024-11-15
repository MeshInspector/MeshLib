#include "MRFloatGrid.h"
#include "MRVDBFloatGrid.h"
#include "MRVDBConversions.h"
#include "MRVDBProgressInterrupter.h"

#include "MRMesh/MRVector3.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRVolumeIndexer.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRBox.h"

namespace MR
{

size_t heapBytes( const FloatGrid& grid )
{
    return grid ? grid->memUsage() : 0;
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
    // for some reason openvdb does not resample correctly for GRID_LEVEL_SET
    auto backupClass = grid_.getGridClass();
    if ( backupClass == openvdb::GRID_LEVEL_SET )
        const_cast< openvdb::FloatGrid& >( grid_ ).setGridClass( openvdb::GRID_FOG_VOLUME );

    // just grows to 100% 
    // first grows fast, then slower
    ProgressCallback dummyProgressCb;
    float i = 1.0f;
    if ( cb )
        dummyProgressCb = [&] ( float )->bool
    {
        i += 1e-4f;
        return cb( 1.0f - 1.0f / std::sqrt( i ) );
    };

    ProgressInterrupter interrupter( dummyProgressCb );
    // openvdb::util::NullInterrupter template argument to avoid tbb inconsistency
    openvdb::tools::resampleToMatch<openvdb::tools::BoxSampler, openvdb::util::NullInterrupter>( grid_, *dest, interrupter );

    // restore original grid class
    if ( backupClass == openvdb::GRID_LEVEL_SET )
        const_cast< openvdb::FloatGrid& >( grid_ ).setGridClass( openvdb::GRID_LEVEL_SET );
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

FloatGrid cropped( const FloatGrid& grid, const Box3i& box, ProgressCallback cb )
{
    if ( !grid )
        return {};
    const openvdb::FloatGrid& grid_ = *grid;
    MR_TIMER;
    openvdb::FloatGrid::Ptr dest = openvdb::FloatGrid::create( grid_.background() );
    dest->setGridClass( grid_.getGridClass() );
    auto dstAcc = dest->getAccessor();
    auto srcAcc = grid_.getConstAccessor();
    size_t pgCounter = 0;
    size_t volume = size_t( width( box ) ) * height( box ) * depth( box );
    for ( int z = box.min.z; z < box.max.z; ++z )
    for ( int y = box.min.y; y < box.max.y; ++y )
    for ( int x = box.min.x; x < box.max.x; ++x )
    {
        openvdb::Coord srcCoord( x, y, z );
        openvdb::Coord dstCoord( x - box.min.x, y - box.min.y, z - box.min.z );
        dstAcc.setValue( dstCoord, srcAcc.getValue( srcCoord ) );
        if ( cb && ( ++pgCounter ) % 1024 == 0 && !cb( float( pgCounter ) / float( volume ) ) )
            return {};
    }
    dest->pruneGrid();
    return MakeFloatGrid( std::move( dest ) );
}

float getValue( const FloatGrid & grid, const Vector3i & p )
{
    return grid ? grid->getConstAccessor().getValue( openvdb::Coord{ p.x, p.y, p.z } ) : 0;
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

void setLevelSetType( FloatGrid & grid )
{
    if ( grid )
        grid->setGridClass( openvdb::GRID_LEVEL_SET );
}

FloatGrid operator += ( FloatGrid & a, const FloatGrid & b )
{
    MR_TIMER
    openvdb::tools::csgUnion( ovdb( *a ), ovdb( *b ) );
    return a;
}

FloatGrid operator -= ( FloatGrid & a, const FloatGrid & b )
{
    MR_TIMER
    openvdb::tools::csgDifference( ovdb( *a ), ovdb( *b ) );
    return a;
}

FloatGrid operator *= ( FloatGrid & a, const FloatGrid & b )
{
    MR_TIMER
    openvdb::tools::csgIntersection( ovdb( *a ), ovdb( *b ) );
    return a;
}

} //namespace MR
