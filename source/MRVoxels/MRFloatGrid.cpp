#include "MRFloatGrid.h"
#include "MRVDBFloatGrid.h"
#include "MRVDBConversions.h"
#include "MRVDBProgressInterrupter.h"

#include "MRMesh/MRVector3.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRVolumeIndexer.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRBox.h"
#include "MRPch/MRSpdlog.h"

namespace MR
{

FloatGrid::FloatGrid() = default;

FloatGrid::FloatGrid( std::shared_ptr<OpenVdbFloatGrid> ptr )
    : ptr_( std::move( ptr ) )
{
}

void FloatGrid::reset() noexcept
{
    ptr_.reset();
}

void FloatGrid::swap( FloatGrid& other ) noexcept
{
    ptr_.swap( other.ptr_ );
}

OpenVdbFloatGrid* FloatGrid::get() const noexcept
{
    return ptr_.get();
}

OpenVdbFloatGrid& FloatGrid::operator*() const noexcept
{
    return *ptr_;
}

OpenVdbFloatGrid* FloatGrid::operator->() const noexcept
{
    return ptr_.get();
}

FloatGrid::operator bool() const noexcept
{
    return (bool)ptr_;
}

std::shared_ptr<OpenVdbFloatGrid> FloatGrid::toVdb() const noexcept
{
    return ptr_;
}

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
    openvdb::FloatGrid::Ptr dest = openvdb::FloatGrid::create( grid->background() );
    dest->setGridClass( grid->getGridClass() );
    openvdb::Mat4R transform;
    transform.setToScale( openvdb::Vec3R{ voxelScale.x,voxelScale.y,voxelScale.z } );
    dest->setTransform( openvdb::math::Transform::createLinearTransform( transform ) ); // org voxel size is 1.0f

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

    // the following piece of code is taken from `openvdb::resampleToMatch` and slightly rewritten to take into account
    // the specific properties of the usage of OpenVdb in our software
    bool failed = true;

    // the level set is processed differently due to the potential narrowness of the band near ISO value. It could be just a 1 voxel, in which case
    // the usual resampling will introduce artifacts and change the topology of ISO surface
    if ( dest->getGridClass() == openvdb::GRID_LEVEL_SET )
    {
        try {
            // unlike `openvdb::resampleToMatch`, the size of the voxel for the grid is always 1, the true size of the voxel is stored
            // in volume wrapper
            dest = openvdb::tools::doLevelSetRebuild( grid_, 0.f, 1, 1, &dest->constTransform(), &interrupter );
            failed = false;
        }
        catch( std::exception& e )
        {
            spdlog::warn( "The input grid is classified as a level set, but it has a value type that is not supported by the level set rebuild tool: {}", e.what() );
        }
    }
    // in case of a volume created in "unsigned mode", which is not supported by OpenVdb as level set but still used as such, the result is empty
    // we detect this case and do resampling
    if ( failed || dest->evalActiveVoxelDim().asVec3I().product() == 0 )
    {
        openvdb::tools::doResampleToMatch<openvdb::tools::BoxSampler, openvdb::util::NullInterrupter>( grid_, *dest, interrupter );
    }

    if ( interrupter.getWasInterrupted() )
        return {};
    // restore normal scale
    dest->setTransform( openvdb::math::Transform::createLinearTransform( 1.0f ) );

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

void gaussianFilter( FloatGrid& grid, int width, int iters, ProgressCallback cb /*= {} */ )
{
    if ( !grid )
        return;
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

    auto filter = openvdb::tools::Filter<openvdb::FloatGrid, openvdb::FloatGrid::ValueConverter<float>::Type, ProgressInterrupter>( ovdb( *grid ), &interrupter );
    filter.gaussian( width, iters );
}

FloatGrid gaussianFiltered( const FloatGrid& grid, int width, int iters, ProgressCallback cb /*= {} */ )
{
    if ( !grid )
        return {};
    auto dest = MakeFloatGrid( grid->deepCopy() );

    gaussianFilter( dest, width, iters, cb );
    return dest;
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
void setValue( FloatGrid& grid, const Vector3i& p, float value )
{
    if ( grid )
        grid->getAccessor().setValue( openvdb::Coord{ p.x,p.y,p.z }, value );
}

void setLevelSetType( FloatGrid & grid )
{
    if ( grid )
        grid->setGridClass( openvdb::GRID_LEVEL_SET );
}

FloatGrid operator += ( FloatGrid & a, const FloatGrid & b )
{
    MR_TIMER;
    openvdb::tools::csgUnion( ovdb( *a ), ovdb( *b ) );
    return a;
}

FloatGrid operator -= ( FloatGrid & a, const FloatGrid & b )
{
    MR_TIMER;
    openvdb::tools::csgDifference( ovdb( *a ), ovdb( *b ) );
    return a;
}

FloatGrid operator *= ( FloatGrid & a, const FloatGrid & b )
{
    MR_TIMER;
    openvdb::tools::csgIntersection( ovdb( *a ), ovdb( *b ) );
    return a;
}

} //namespace MR
