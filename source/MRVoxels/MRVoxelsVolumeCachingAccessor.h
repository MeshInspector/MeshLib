#pragma once

#include "MRVoxelsVolumeAccess.h"
#include "MRMesh/MRTimer.h"

namespace MR
{

/// This accessor first loads data for given number of layers in internal cache, and then returns values from the cache.
/// Direct access to data outside of cache is not allowed.
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
        , layers_( params_.preloadedLayerCount )
        , firstLayerVoxelId_( params_.preloadedLayerCount )
    {
        assert( params_.preloadedLayerCount > 0 );
        for ( auto & l : layers_ )
            l.resize( indexer_.sizeXY() );
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
        for ( auto i = 0; i + 1 < layers_.size(); ++i )
        {
            std::swap( layers_[i], layers_[i + 1] );
            firstLayerVoxelId_[i] = firstLayerVoxelId_[i + 1];
        }
        if ( z_ + params_.preloadedLayerCount - 1 < indexer_.dims().z )
            preloadLayer_( params_.preloadedLayerCount - 1 );
    }

    /// get voxel volume data
    ValueType get( const VoxelLocation & loc ) const
    {
        const auto layerIndex = loc.pos.z - z_;
        assert( 0 <= layerIndex && layerIndex < layers_.size() );
        assert( loc.id >= firstLayerVoxelId_[layerIndex] );
        assert( loc.id < firstLayerVoxelId_[layerIndex] + indexer_.sizeXY() );
        return layers_[layerIndex][loc.id - firstLayerVoxelId_[layerIndex]];
    }

private:
    [[nodiscard]] size_t toLayerIndex( const Vector3i& pos ) const
    {
        return indexer_.toVoxelId( { pos.x, pos.y, 0 } );
    }

    void preloadLayer_( size_t layerIndex )
    {
        MR_TIMER
        assert( layerIndex < layers_.size() );
        auto& layer = layers_[layerIndex];
        const auto z = z_ + (int)layerIndex;
        const auto& dims = indexer_.dims();
        assert( 0 <= z && z < dims.z );
        auto loc = indexer_.toLoc( Vector3i{ 0, 0, z } );
        firstLayerVoxelId_[layerIndex] = loc.id;
        size_t n = 0;
        for ( loc.pos.y = 0; loc.pos.y < dims.y; ++loc.pos.y )
            for ( loc.pos.x = 0; loc.pos.x < dims.x; ++loc.pos.x, ++loc.id, ++n )
                layer[n] = accessor_.get( loc );
    }

private:
    const VoxelsVolumeAccessor<V>& accessor_;
    VolumeIndexer indexer_;
    Parameters params_;

    int z_ = -1;
    std::vector<std::vector<ValueType>> layers_;
    std::vector<VoxelId> firstLayerVoxelId_;
};

} // namespace MR

