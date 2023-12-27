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
    using ValueType = typename VolumeType::ValueType;

    explicit VoxelsVolumeAccessor( const VolumeType& volume )
        : accessor_( volume.data->getConstAccessor() )
        , minCoord_( volume.data->evalActiveVoxelBoundingBox().min() )
    {}

    ValueType get( const Vector3i& pos ) const
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
    using ValueType = typename VolumeType::ValueType;

    explicit VoxelsVolumeAccessor( const VolumeType& volume )
        : data_( volume.data )
          , indexer_( volume.dims )
    {}

    ValueType get( const Vector3i& pos ) const
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
    using ValueType = typename VolumeType::ValueType;

    explicit VoxelsVolumeAccessor( const VolumeType& volume )
        : data_( volume.data )
    {}

    ValueType get( const Vector3i& pos ) const
    {
        return data_( pos );
    }

private:
    const VoxelValueGetter<T>& data_;
};

/// helper class to preload voxel volume data
template <typename V>
class VoxelsVolumeCachingAccessor
{
public:
    using VolumeType = V;
    using ValueType = typename V::ValueType;

    struct Parameters
    {
        /// amount of layers to be preloaded
        size_t preloadedLayerCount = 1;
    };

    VoxelsVolumeCachingAccessor( const VoxelsVolumeAccessor<V>& accessor, const VolumeIndexer& indexer, Parameters parameters = {} )
        : accessor_( accessor )
        , indexer_( indexer )
        , params_( std::move( parameters ) )
        , layers_( params_.preloadedLayerCount, std::vector<ValueType>( indexer_.sizeXY() ) )
    {
        assert( params_.preloadedLayerCount > 0 );
    }

    /// get current layer
    [[nodiscard]] int currentLayer() const
    {
        return z_;
    }

    /// preload layers, starting from z
    void preloadLayer( int z )
    {
        assert( 0 <= z && z < indexer_.dims().z );
        z_ = z;
        for ( auto layerIndex = 0; layerIndex < layers_.size(); ++layerIndex )
        {
            if ( indexer_.dims().z <= z_ + layerIndex )
                break;
            preloadLayer_( layerIndex );
        }
    }

    /// preload the next layer
    void preloadNextLayer()
    {
        z_ += 1;
        for ( auto i = 0, j = 1; j < layers_.size(); ++i, ++j )
            std::swap( layers_[i], layers_[j] );
        if ( z_ + params_.preloadedLayerCount - 1 < indexer_.dims().z )
            preloadLayer_( params_.preloadedLayerCount - 1 );
    }

    /// get voxel volume data
    ValueType get( const Vector3i& pos ) const
    {
        const auto layerIndex = pos.z - z_;
        if ( 0 <= layerIndex && layerIndex < layers_.size() )
            return layers_[layerIndex][toLayerIndex( pos )];

        return accessor_.get( pos );
    }

private:
    [[nodiscard]] size_t toLayerIndex( const Vector3i& pos ) const
    {
        return indexer_.toVoxelId( { pos.x, pos.y, 0 } );
    }

    void preloadLayer_( size_t layerIndex )
    {
        assert( layerIndex < layers_.size() );
        auto& layer = layers_[layerIndex];
        const auto z = z_ + (int)layerIndex;
        const auto& dims = indexer_.dims();
        assert( 0 <= z && z < dims.z );
        Vector3i pos { 0, 0, z };
        for ( pos.y = 0; pos.y < dims.y; ++pos.y )
            for ( pos.x = 0; pos.x < dims.x; ++pos.x )
                layer[toLayerIndex( pos )] = accessor_.get( pos );
    }

private:
    const VoxelsVolumeAccessor<V>& accessor_;
    VolumeIndexer indexer_;
    Parameters params_;

    int z_ = -1;
    std::vector<std::vector<ValueType>> layers_;
};

} // namespace MR
