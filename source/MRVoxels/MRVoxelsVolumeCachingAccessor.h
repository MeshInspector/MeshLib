#pragma once

#include "MRVoxelsVolumeAccess.h"
#include "MRMesh/MRParallelFor.h"
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

    /// preload layers, starting from z;
    /// return false if the operation was cancelled from callback
    bool preloadLayer( int z, const ProgressCallback& cb = {} )
    {
        assert( 0 <= z && z < indexer_.dims().z );
        z_ = z;
        for ( size_t layerIndex = 0; layerIndex < layers_.size(); ++layerIndex )
        {
            if ( indexer_.dims().z <= z_ + layerIndex )
                break;
            if ( !preloadLayer_( layerIndex, subprogress( cb, layerIndex, layers_.size() ) ) )
                return false;
        }
        return true;
    }

    /// preload the next layer;
    /// return false if the operation was cancelled from callback
    bool preloadNextLayer( const ProgressCallback& cb = {} )
    {
        z_ += 1;
        for ( auto i = 0; i + 1 < layers_.size(); ++i )
        {
            std::swap( layers_[i], layers_[i + 1] );
            firstLayerVoxelId_[i] = firstLayerVoxelId_[i + 1];
        }
        if ( z_ + params_.preloadedLayerCount - 1 < indexer_.dims().z )
            return preloadLayer_( params_.preloadedLayerCount - 1, cb );
        return true;
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

    bool preloadLayer_( size_t layerIndex, const ProgressCallback& cb )
    {
        MR_TIMER
        assert( layerIndex < layers_.size() );
        auto& layer = layers_[layerIndex];
        const auto z = z_ + (int)layerIndex;
        const auto& dims = indexer_.dims();
        assert( 0 <= z && z < dims.z );
        firstLayerVoxelId_[layerIndex] = indexer_.toVoxelId( Vector3i{ 0, 0, z } );
        return ParallelFor( 0, dims.y, [&]( int y )
        {
            auto accessor = accessor_; // only for OpenVDB accessor, which is not thread-safe
            auto loc = indexer_.toLoc( Vector3i{ 0, y, z } );
            size_t n = size_t( y ) * dims.x;
            for ( loc.pos.x = 0; loc.pos.x < dims.x; ++loc.pos.x, ++loc.id, ++n )
                layer[n] = accessor.get( loc );
        }, cb, 1 );
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

