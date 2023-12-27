#pragma once

#include "MRMeshFwd.h"
#include "MRSimpleVolume.h"
#include "MRVolumeIndexer.h"

#ifndef MRMESH_NO_VOXEL
#include "MRVDBFloatGrid.h"
#endif

namespace MR
{

/// helper class for generalized voxel volume data access
template <typename Volume>
class VoxelsVolumeAccessor;

#ifndef MRMESH_NO_VOXEL
template <>
class VoxelsVolumeAccessor<VdbVolume>
{
public:
    using VolumeType = VdbVolume;

    explicit VoxelsVolumeAccessor( const VolumeType& volume )
        : accessor_( volume.data->getConstAccessor() )
        , minCoord_( volume.data->evalActiveVoxelBoundingBox().min() )
    {}

    VolumeType::ValueType get( const Vector3i& pos ) const
    {
        const openvdb::Coord coord {
            pos.x + minCoord_.x(),
            pos.y + minCoord_.y(),
            pos.z + minCoord_.z(),
        };
        return accessor_.getValue( coord );
    }

private:
    openvdb::FloatGrid::ConstAccessor accessor_;
    openvdb::Coord minCoord_;
};
#endif

template <typename T>
class VoxelsVolumeAccessor<VoxelsVolume<std::vector<T>>>
{
public:
    using VolumeType = VoxelsVolume<std::vector<T>>;

    explicit VoxelsVolumeAccessor( const VolumeType& volume )
        : data_( volume.data )
          , indexer_( volume.dims )
    {}

    VolumeType::ValueType get( const Vector3i& pos ) const
    {
        return data_[indexer_.toVoxelId( pos )];
    }

private:
    const std::vector<T>& data_;
    VolumeIndexer indexer_;
};

template <typename T>
class VoxelsVolumeAccessor<VoxelsVolume<VoxelValueGetter<T>>>
{
public:
    using VolumeType = VoxelsVolume<VoxelValueGetter<T>>;

    explicit VoxelsVolumeAccessor( const VolumeType& volume )
        : data_( volume.data )
    {}

    VolumeType::ValueType get( const Vector3i& pos ) const
    {
        return data_( pos );
    }

private:
    const VoxelValueGetter<T>& data_;
};

} // namespace MR
