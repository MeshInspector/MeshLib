#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "MRVolumeSegment.h"
#include "MRVoxelPath.h"
#include "MRSimpleVolume.h"
#include "MRVolumeIndexer.h"
#include "MRBox.h"
#include "MRVoxelGraphCut.h"
#include "MRVDBConversions.h"
#include "MRFloatGrid.h"
#include "MRMesh.h"
#include <filesystem>

namespace MR
{

// creates mesh from simple volume as 0.5 iso-surface
tl::expected<MR::Mesh, std::string> meshFromSimpleVolume( const SimpleVolume& volumePart, const Vector3i& shift )
{
    auto grid = simpleVolumeToDenseGrid( volumePart );
    auto mesh = gridToMesh( std::move( grid ), volumePart.voxelSize, 0.5f ).value(); // no callback so cannot be stopped

    auto minCorner = mult( Vector3f( shift ), volumePart.voxelSize );
    for ( auto& p : mesh.points )
    {
        p = p + minCorner;
    }

    if ( mesh.topology.numValidFaces() == 0 )
        return tl::make_unexpected( "Failed to create mesh from mask" );

    return mesh;
}

// returns block of volume and new mask in space of new block, and shift of block
std::tuple<SimpleVolume, VoxelBitSet, Vector3i> simpleVolumeFromVoxelsMask( const VdbVolume& volume, const VoxelBitSet& mask, int expansion )
{
    assert( volume.data );
    assert( mask.any() );

    SimpleVolume volumePart;
    volumePart.voxelSize = volume.voxelSize;
    auto expandedMask = mask;
    const auto& indexer = VolumeIndexer( volume.dims );
    expandVoxelsMask( expandedMask, indexer, expansion );

    Box3i partBox;
    for ( auto voxelId : expandedMask )
    {
        auto pos = indexer.toPos( voxelId );
        partBox.include( pos );
    }

    volumePart.dims = partBox.size() + Vector3i::diagonal( 1 );
    const int newDimX = volumePart.dims.x;
    const int newDimXY = newDimX * volumePart.dims.y;
    volumePart.data.resize( newDimXY * volumePart.dims.z );

    auto partIndexer = VolumeIndexer( volumePart.dims );
    VoxelBitSet volumePartMask( volumePart.data.size() );
    auto accessor = volume.data->getConstAccessor();
    for ( VoxelId i = VoxelId( 0 ); i < volumePart.data.size(); ++i )
    {
        auto pos = partIndexer.toPos( i ) + partBox.min;
        if ( mask.test( indexer.toVoxelId( pos ) ) )
            volumePartMask.set( i );
        volumePart.data[i] = accessor.getValue( { pos.x,pos.y,pos.z } );
    }
    return { volumePart,volumePartMask,partBox.min };
}

// Mode of meshing
enum class VolumeMaskMeshingMode
{
    Simple, // 1 inside, 0 outside, 0.5 iso
    Smooth // 1 deep inside, (density - outsideAvgDensity)/(insideAvgDensity - outsideAvgDensity) on the edge, 0 - far outside, 0.5 iso
};

// changes volume part due to meshing mode
void prepareVolumePart( SimpleVolume& volumePart, const VoxelBitSet& mask, VolumeMaskMeshingMode mode )
{
    if ( mode == VolumeMaskMeshingMode::Simple )
    {
        for ( VoxelId i = VoxelId( 0 ); i < volumePart.data.size(); ++i )
        {
            volumePart.data[i] = mask.test( i ) ? 1.0f : 0.0f;
        }
        return;
    }
    // else: mode == VolumeMaskMeshingMode::Smooth
    double insideAvg = 0.0;
    double outsideAvg = 0.0;
    for ( VoxelId i = VoxelId( 0 ); i < volumePart.data.size(); ++i )
    {
        if ( mask.test( i ) )
            insideAvg += volumePart.data[i];
        else
            outsideAvg += volumePart.data[i];
    }
    insideAvg /= double( mask.count() );
    outsideAvg /= double( volumePart.data.size() - mask.count() );
    auto range = float( insideAvg - outsideAvg );

    auto partIndexer = VolumeIndexer( volumePart.dims );
    auto smallExpMask = mask;
    auto smallShrMask = mask;
    expandVoxelsMask( smallExpMask, partIndexer, 3 );
    shrinkVoxelsMask( smallShrMask, partIndexer, 3 );
    for ( VoxelId i = VoxelId( 0 ); i < volumePart.data.size(); ++i )
    {
        if ( smallShrMask.test( i ) )
            volumePart.data[i] = 1.0f;
        else if ( smallExpMask.test( i ) )
            volumePart.data[i] = std::clamp( float( volumePart.data[i] - outsideAvg ) / range, 0.0f, 1.0f );
        else
            volumePart.data[i] = 0.0f;
    }
}

tl::expected<MR::Mesh, std::string> meshFromVoxelsMask( const VdbVolume& volume, const VoxelBitSet& mask )
{
    if ( !volume.data )
        return tl::make_unexpected( "Cannot create mesh from empty volume." );
    if ( mask.none() )
        return tl::make_unexpected( "Cannot create mesh from empty mask." );

    auto [volumePart, partMask, minVoxel] = simpleVolumeFromVoxelsMask( volume, mask, 25 );

    prepareVolumePart( volumePart, partMask, VolumeMaskMeshingMode::Smooth );

    return meshFromSimpleVolume( volumePart, minVoxel );
}

tl::expected<MR::Mesh, std::string> segmentVolume( const VdbVolume& volume, const std::vector<std::pair<Vector3f, Vector3f>>& pairs,
                                                   const VolumeSegmentationParameters& params )
{
    VolumeSegmenter segmentator( volume );
    VolumeIndexer indexer( volume.dims );
    Vector3f reverseVoxelsSize( 1.f / volume.voxelSize.x, 1.f / volume.voxelSize.y, 1.f / volume.voxelSize.z );
    for ( const auto& [start, stop] : pairs )
    {
        VoxelMetricParameters metricParams;
        metricParams.start = size_t( indexer.toVoxelId( Vector3i( mult( start, reverseVoxelsSize ) ) ) );
        metricParams.stop = size_t( indexer.toVoxelId( Vector3i( mult( stop, reverseVoxelsSize ) ) ) );
        constexpr char mask = 1;
        for ( int i = 0; i < 4; ++i )
        {
            metricParams.quatersMask = mask << i;
            segmentator.addPathSeeds( metricParams, VolumeSegmenter::SeedType::Inside, params.buildPathExponentModifier );
        }
    }
    auto segmentation = segmentator.segmentVolume( params.segmentationExponentModifier, params.voxelsExpansion );
    if ( !segmentation.has_value() )
        return tl::make_unexpected( segmentation.error() );
    return segmentator.createMeshFromSegmentation( segmentation.value() );
}

// Class implementation

VolumeSegmenter::VolumeSegmenter( const VdbVolume& volume ):
    volume_{volume}
{
}

void VolumeSegmenter::addPathSeeds( const VoxelMetricParameters& metricParameters, SeedType seedType, float exponentModifier /*= -1.0f */ )
{
    auto metric = voxelsExponentMetric( volume_, metricParameters, exponentModifier );
    auto path = buildSmallestMetricPath( volume_, metric, metricParameters.start, metricParameters.stop );

    auto& curSeeds = seeds_[seedType];
    auto shift = curSeeds.size();
    curSeeds.resize( shift + path.size() );
    VolumeIndexer indexer( volume_.dims );
    for ( int p = 0; p < path.size(); ++p )
    {
        curSeeds[shift + p] = indexer.toPos( VoxelId( int( path[p] ) ) );
    }
    seedsChanged_ = true;
}

void VolumeSegmenter::setSeeds( const std::vector<Vector3i>& seeds, SeedType seedType )
{
    seeds_[seedType] = seeds;
    seedsChanged_ = true;
}

void VolumeSegmenter::addSeeds( const std::vector<Vector3i>& seeds, SeedType seedType )
{
    auto& curSeeds = seeds_[seedType];
    curSeeds.reserve( curSeeds.size() + seeds.size() );
    curSeeds.insert( curSeeds.end(), seeds.begin(), seeds.end() );
    seedsChanged_ = true;
}

const std::vector<MR::Vector3i>& VolumeSegmenter::getSeeds( SeedType seedType ) const
{
    return seeds_[seedType];
}

tl::expected<VoxelBitSet, std::string> VolumeSegmenter::segmentVolume( float segmentationExponentModifier /*= 3000.0f*/, int voxelsExpansion /*= 25 */, ProgressCallback cb /* =nullptr */)
{
    if ( seeds_[Inside].empty() )
        return tl::make_unexpected( "No seeds presented" );

    if ( !volume_.data )
        return tl::make_unexpected( "Volume contain no grid" );

    if ( seedsChanged_ )
    {
        setupVolumePart_( voxelsExpansion );
        seedsChanged_ = false;
    }

    // Segment volume
    return segmentVolumeByGraphCut( volumePart_, segmentationExponentModifier, seedsInVolumePartSpace_[Inside], seedsInVolumePartSpace_[Outside], cb );
}

tl::expected<MR::Mesh, std::string> VolumeSegmenter::createMeshFromSegmentation( const VoxelBitSet& segmentation ) const
{
    auto segmentBlockCopy = volumePart_;
    segmentBlockCopy.voxelSize = volume_.voxelSize;
    prepareVolumePart( segmentBlockCopy, segmentation, VolumeMaskMeshingMode::Simple );
    return meshFromSimpleVolume( segmentBlockCopy, minVoxel_ );
}

const MR::Vector3i& VolumeSegmenter::getVolumePartDimensions() const
{
    return volumePart_.dims;
}

const Vector3i& VolumeSegmenter::getMinVoxel() const
{
    return minVoxel_;
}

void VolumeSegmenter::setupVolumePart_( int voxelsExpansion )
{
    auto& curSeeds = seeds_[Inside];
    auto minmaxElemX = std::minmax_element( curSeeds.begin(), curSeeds.end(), []( const Vector3i& first, const Vector3i& second )
    {
        return first.x < second.x;
    } );

    auto minmaxElemY = std::minmax_element( curSeeds.begin(), curSeeds.end(), []( const Vector3i& first, const Vector3i& second )
    {
        return first.y < second.y;
    } );

    auto minmaxElemZ = std::minmax_element( curSeeds.begin(), curSeeds.end(), []( const Vector3i& first, const Vector3i& second )
    {
        return first.z < second.z;
    } );

    auto minVoxel = Vector3i( minmaxElemX.first->x, minmaxElemY.first->y, minmaxElemZ.first->z );
    auto maxVoxel = Vector3i( minmaxElemX.second->x, minmaxElemY.second->y, minmaxElemZ.second->z );

    // need to fix dims: clamp by real voxels bounds
    maxVoxel += Vector3i::diagonal( voxelsExpansion );
    minVoxel -= Vector3i::diagonal( voxelsExpansion );


    const auto& dims = volume_.dims;

    maxVoxel.x = std::min( maxVoxel.x, dims.x );
    maxVoxel.y = std::min( maxVoxel.y, dims.y );
    maxVoxel.z = std::min( maxVoxel.z, dims.z );

    minVoxel.x = std::max( minVoxel.x, 0 );
    minVoxel.y = std::max( minVoxel.y, 0 );
    minVoxel.z = std::max( minVoxel.z, 0 );

    bool blockChanged{false};
    if ( minVoxel != minVoxel_ )
    {
        minVoxel_ = minVoxel;
        blockChanged = true;
    }
    if ( maxVoxel != maxVoxel_ )
    {
        maxVoxel_ = maxVoxel;
        blockChanged = true;
    }

    if ( blockChanged )
    {
        volumePart_.dims = maxVoxel - minVoxel + Vector3i::diagonal( 1 );
        const int newDimX = volumePart_.dims.x;
        const size_t newDimXY = size_t( newDimX ) * volumePart_.dims.y;
        volumePart_.data.resize( newDimXY * volumePart_.dims.z );
        const auto& accessor = volume_.data->getConstAccessor();
        for ( int z = minVoxel.z; z <= maxVoxel.z; ++z )
            for ( int y = minVoxel.y; y <= maxVoxel.y; ++y )
                for ( int x = minVoxel.x; x <= maxVoxel.x; ++x )
                {
                    volumePart_.data[x - minVoxel.x + ( y - minVoxel.y ) * newDimX + ( z - minVoxel.z ) * newDimXY] =
                        accessor.getValue( {x,y,z} );
                }

        auto minmaxIt = std::minmax_element( volumePart_.data.begin(), volumePart_.data.end() );
        volumePart_.min = *minmaxIt.first;
        volumePart_.max = *minmaxIt.second;


        seedsInVolumePartSpace_[Inside].resize( newDimXY * volumePart_.dims.z );
        seedsInVolumePartSpace_[Outside].resize( newDimXY * volumePart_.dims.z );
    }
    seedsInVolumePartSpace_[Inside].reset();
    seedsInVolumePartSpace_[Outside].reset();

    const int newDimX = volumePart_.dims.x;
    const size_t newDimXY = size_t( newDimX ) * volumePart_.dims.y;

    auto CoordToNewVoxelId = [&]( const Vector3i& coord )->VoxelId
    {
        return VoxelId( coord.x + coord.y * newDimX + coord.z * newDimXY );
    };

    for ( const auto& seed : seeds_[Inside] )
    {
        seedsInVolumePartSpace_[Inside].set( CoordToNewVoxelId( seed - minVoxel_ ) );
    }
    for ( auto seed : seeds_[Outside] )
    {
        seed.x = std::clamp( seed.x, minVoxel_.x, maxVoxel_.x );
        seed.y = std::clamp( seed.y, minVoxel_.y, maxVoxel_.y );
        seed.z = std::clamp( seed.z, minVoxel_.z, maxVoxel_.z );
        seedsInVolumePartSpace_[Outside].set( CoordToNewVoxelId( seed - minVoxel_ ) );
    }
    for ( int i = 0; i < 3; ++i ) // fill non tooth voxels by faces of segment block
    {
        int axis1 = ( i + 1 ) % 3;
        int axis2 = ( i + 2 ) % 3;
        for ( int a1 = 0; a1 < volumePart_.dims[axis1]; ++a1 )
            for ( int a2 = 0; a2 < volumePart_.dims[axis2]; ++a2 )
            {
                Vector3i nearVoxel, farVoxel;
                nearVoxel[i] = 0; nearVoxel[axis1] = a1; nearVoxel[axis2] = a2;
                farVoxel = nearVoxel; farVoxel[i] = volumePart_.dims[i] - 1;
                seedsInVolumePartSpace_[Outside].set( CoordToNewVoxelId( nearVoxel ) );
                seedsInVolumePartSpace_[Outside].set( CoordToNewVoxelId( farVoxel ) );
            }
    }

    seedsInVolumePartSpace_[Outside] -= seedsInVolumePartSpace_[Inside];
}

}
#endif
