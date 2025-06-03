#pragma once

#include "MRVoxelsVolumeAccess.h"

namespace MR
{

/// helper class for generalized access to voxel volume data with trilinear interpolation
/// coordinate: 0       voxelSize
///             |       |
///             I---*---I---*---I---
///             |       |       |
/// value:     [0]     [1]     [2] ...
/// note: this class is as thread-safe as the underlying Accessor
/// e.g. VoxelsVolumeAccessor<VdbVolume> is not thread-safe (but several instances on same volume is thread-safe)
template <typename Accessor>
class VoxelsVolumeInterpolatedAccessor
{
public:
    using VolumeType = typename Accessor::VolumeType;
    using ValueType = typename Accessor::ValueType;

    /// create an accessor instance that stores references to volume and its accessor
    /// the volume should not modified while it is accessed by this class
    explicit VoxelsVolumeInterpolatedAccessor( const VolumeType& volume, const Accessor& accessor )
        : volume_( volume ), accessor_( accessor )
    {}

    /// delete copying constructor to avoid accidentally creating non-thread-safe accessors
    VoxelsVolumeInterpolatedAccessor( const VoxelsVolumeInterpolatedAccessor& ) = delete;
    /// a copying-like constructor with explicitly provided accessor
    explicit VoxelsVolumeInterpolatedAccessor( const VoxelsVolumeInterpolatedAccessor& other, const Accessor& accessor )
        : volume_( other.volume_ ), accessor_( accessor )
    {}

    /// get value at specified coordinates
    ValueType get( const Vector3f& pos ) const
    {
        IndexAndPos index = getIndexAndPos( pos - mult( accessor_.shift(), volume_.voxelSize ) );
        ValueType value{};
        float cx[2] = { 1.0f - index.pos.x, index.pos.x };
        float cy[2] = { 1.0f - index.pos.y, index.pos.y };
        float cz[2] = { 1.0f - index.pos.z, index.pos.z };
        for ( int i = 0; i < 8; i++ )
        {
            Vector3i d{ i & 1, ( i >> 1 ) & 1, i >> 2 };
            const auto voxPos = index.index + d;
            if ( voxPos.x >= 0 && voxPos.x < volume_.dims.x &&
                 voxPos.y >= 0 && voxPos.y < volume_.dims.y &&
                 voxPos.z >= 0 && voxPos.z < volume_.dims.z )
            {
                value += accessor_.get( voxPos ) * ( cx[d.x] * cy[d.y] * cz[d.z] );
            }
        }
        return value;
    }

private:
    const VolumeType& volume_;
    const Accessor& accessor_;

    struct IndexAndPos
    {
        Vector3i index; // Zero-based voxel index in the volume
        Vector3f pos;   // [0;1) position of a point in the voxel: 0 corresponds to index and 1 to next voxel
    };

    IndexAndPos getIndexAndPos( Vector3f pos ) const
    {
        IndexAndPos res;
        res.pos.x = pos.x / volume_.voxelSize.x;
        res.pos.y = pos.y / volume_.voxelSize.y;
        res.pos.z = pos.z / volume_.voxelSize.z;
        pos.x = floor( res.pos.x );
        pos.y = floor( res.pos.y );
        pos.z = floor( res.pos.z );
        res.index.x = int( pos.x );
        res.index.y = int( pos.y );
        res.index.z = int( pos.z );
        res.pos.x -= pos.x;
        res.pos.y -= pos.y;
        res.pos.z -= pos.z;
        return res;
    }
};

/// sample function that resamples the voxel volume using interpolating accessor
template <typename Accessor>
SimpleVolumeMinMax resampleVolumeByInterpolation(
    const typename Accessor::VolumeType &volume,
    const Accessor &accessor,
    const Vector3f &newVoxelSize )
{
    SimpleVolumeMinMax res{
        { .voxelSize{ newVoxelSize } },
        { volume.min, volume.max }
    };
    res.dims.x = int( volume.dims.x * volume.voxelSize.x / res.voxelSize.x );
    res.dims.y = int( volume.dims.y * volume.voxelSize.y / res.voxelSize.y );
    res.dims.z = int( volume.dims.z * volume.voxelSize.z / res.voxelSize.z );
    res.data.resize( size_t( res.dims.x ) * res.dims.y * res.dims.z );
    VolumeIndexer indexer( res.dims );

    VoxelsVolumeInterpolatedAccessor<Accessor> interpolator( volume, accessor );
    for ( int k = 0; k < res.dims.z; k++ )
    for ( int j = 0; j < res.dims.y; j++ )
    for ( int i = 0; i < res.dims.x; i++ )
        res.data[indexer.toVoxelId( { i, j, k } )] = interpolator.get(
            { i * res.voxelSize.x, j * res.voxelSize.y, k * res.voxelSize.z } );

    return res;
}

} // namespace MR
