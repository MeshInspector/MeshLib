#pragma once

#include "MRVoxelsFwd.h"
#include "MRVoxelsVolume.h"
#include "MRVDBFloatGrid.h"
#include "MRMesh/MRVolumeIndexer.h"
#include "MRMesh/MRIsNaN.h"

namespace MR
{

/// helper class for generalized voxel volume data access
template <typename Volume>
class VoxelsVolumeAccessor;

/// VoxelsVolumeAccessor specialization for VDB volume
template <>
class VoxelsVolumeAccessor<VdbVolume>
{
public:
    using VolumeType = VdbVolume;
    using ValueType = typename VolumeType::ValueType;
    static constexpr bool cacheEffective = true; ///< caching results of this accessor can improve performance

    explicit VoxelsVolumeAccessor( const VolumeType& volume )
        : accessor_( volume.data->getConstAccessor() )
        , minCoord_( fromVdb( volume.data->evalActiveVoxelBoundingBox().min() ) )
    {}

    ValueType get( const Vector3i& pos ) const
    {
        ValueType res;
        if ( !accessor_.probeValue( toVdb( pos + minCoord_ ), res ) )
            return cQuietNan;
        return res;
    }

    ValueType get( const VoxelLocation & loc ) const
    {
        return get( loc.pos );
    }

    const Vector3i& minCoord() const { return minCoord_; }

    /// this additional shift shall be added to integer voxel coordinates during transformation in 3D space
    Vector3f shift() const { return Vector3f( minCoord_ ); }

private:
    openvdb::FloatGrid::ConstAccessor accessor_;
    Vector3i minCoord_;
};

/// VoxelsVolumeAccessor specialization for simple volumes
template <typename T>
class VoxelsVolumeAccessor<VoxelsVolume<Vector<T,VoxelId>>>
{
public:
    using VolumeType = VoxelsVolume<Vector<T,VoxelId>>;
    using ValueType = typename VolumeType::ValueType;
    static constexpr bool cacheEffective = false; ///< caching results of this accessor does not make any sense since it returns values from a simple container

    explicit VoxelsVolumeAccessor( const VolumeType& volume )
        : data_( volume.data )
          , indexer_( volume.dims )
    {}

    ValueType get( const Vector3i& pos ) const
    {
        return data_[indexer_.toVoxelId( pos )];
    }

    ValueType get( const VoxelLocation & loc ) const
    {
        return data_[loc.id];
    }

    /// this additional shift shall be added to integer voxel coordinates during transformation in 3D space
    Vector3f shift() const { return Vector3f::diagonal( 0.5f ); }

private:
    const Vector<T,VoxelId>& data_;
    VolumeIndexer indexer_;
};

/// VoxelsVolumeAccessor specialization for simple volumes with min/max
template <typename T>
class VoxelsVolumeAccessor<VoxelsVolumeMinMax<Vector<T,VoxelId>>> : public VoxelsVolumeAccessor<VoxelsVolume<Vector<T,VoxelId>>>
{
    using Base = VoxelsVolumeAccessor<VoxelsVolume<Vector<T,VoxelId>>>;
public:
    using VolumeType = VoxelsVolumeMinMax<Vector<T,VoxelId>>;
    using ValueType = typename VolumeType::ValueType;
    using Base::cacheEffective;
    using Base::Base;
};

/// VoxelsVolumeAccessor specialization for value getters
template <typename T>
class VoxelsVolumeAccessor<VoxelsVolume<VoxelValueGetter<T>>>
{
public:
    using VolumeType = VoxelsVolume<VoxelValueGetter<T>>;
    using ValueType = typename VolumeType::ValueType;
    static constexpr bool cacheEffective = true; ///< caching results of this accessor can improve performance

    explicit VoxelsVolumeAccessor( const VolumeType& volume )
        : data_( volume.data )
    {}

    ValueType get( const Vector3i& pos ) const
    {
        return data_( pos );
    }

    ValueType get( const VoxelLocation & loc ) const
    {
        return get( loc.pos );
    }

    /// this additional shift shall be added to integer voxel coordinates during transformation in 3D space
    Vector3f shift() const { return Vector3f::diagonal( 0.5f ); }

private:
    const VoxelValueGetter<T>& data_;
};

} // namespace MR
