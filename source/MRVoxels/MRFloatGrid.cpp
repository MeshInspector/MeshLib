#include "MRFloatGrid.h"
#include "MRVDBFloatGrid.h"

#include "MRMesh/MRVector3.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRVolumeIndexer.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRBox.h"

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

FloatGrid FloatGrid::deepCopy( const FloatGrid& other ) noexcept
{
    if ( other )
        return MakeFloatGrid( other->deepCopy() );
    return other;
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

float getValue( const FloatGrid & grid, const Vector3i & p )
{
    return grid ? grid->getConstAccessor().getValue( openvdb::Coord{ p.x, p.y, p.z } ) : 0;
}

void setValue( FloatGrid& grid, const Vector3i& p, float value )
{
    if ( grid )
        grid->getAccessor().setValue( openvdb::Coord{ p.x,p.y,p.z }, value );
}

Box3i findActiveBounds( const FloatGrid& grid )
{
    if ( !grid )
    {
        assert( false );
        return Box3i();
    }
    return fromVdbBox( grid->evalActiveVoxelBoundingBox() );
}

void setValue( FloatGrid & grid, const VoxelBitSet& region, float value )
{
    if ( !grid )
        return;
    MR_TIMER;
    auto bbox = findActiveBounds( grid );
    Vector3i dims = bbox.size();
    VolumeIndexer indexer = VolumeIndexer( dims );
    auto minVox = bbox.min;
    auto accessor = grid->getAccessor();
    for ( auto voxid : region )
    {
        auto pos = indexer.toPos( voxid );
        accessor.setValue( toVdb( minVox + pos ), value );
    }
}

void setValues( FloatGrid& grid, const VoxelBitSet& region, const std::vector<float>& values )
{
    if ( !grid )
        return;
    MR_TIMER;
    auto bbox = findActiveBounds( grid );
    Vector3i dims = bbox.size();
    VolumeIndexer indexer = VolumeIndexer( dims );
    auto minVox = bbox.min;
    auto accessor = grid->getAccessor();
    size_t i = 0;
    for ( auto voxid : region )
    {
        if ( i >= values.size() )
        {
            assert( false );
            return;
        }
        auto pos = indexer.toPos( voxid );
        accessor.setValue( toVdb( minVox + pos ), values[i] );
        ++i;
    }
}

void setLevelSetType( FloatGrid & grid )
{
    if ( grid )
        grid->setGridClass( openvdb::GRID_LEVEL_SET );
}

} //namespace MR
