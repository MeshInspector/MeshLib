#include "MRVDBConversions.h"

#include "MRVDBFloatGrid.h"
#include "MROpenVDB.h"
#include "MRVoxelsVolume.h"
#include "MRMesh/MRVolumeIndexer.h"
#include "MRMesh/MRParallelFor.h"
#include "MRMesh/MRTimer.h"

namespace MR
{

// make VoxelsVolume (e.g. SimpleVolume or SimpleVolumeU16) from VdbVolume
// if VoxelsVolume values type is integral, performs mapping from the sourceScale to
// nonnegative range of target type
template<typename T, bool Norm>
Expected<VoxelsVolumeMinMax<Vector<T,VoxelId>>> vdbVolumeToSimpleVolumeImpl(
    const VdbVolume& vdbVolume, const Box3i& activeBox = Box3i(), std::optional<MinMaxf> maybeSourceScale = {}, ProgressCallback cb = {} )
{
    MR_TIMER;
    constexpr bool isFloat = std::is_same_v<float, T> || std::is_same_v<double, T> || std::is_same_v<long double, T>;

    VoxelsVolumeMinMax<Vector<T,VoxelId>> res;

    res.dims = !activeBox.valid() ? vdbVolume.dims : activeBox.size();
    Vector3i org = activeBox.valid() ? activeBox.min : Vector3i{};
    res.voxelSize = vdbVolume.voxelSize;
    [[maybe_unused]] const auto sourceScale = maybeSourceScale.value_or( MinMaxf{ vdbVolume.min, vdbVolume.max } );
    float targetMin = sourceScale.min, targetMax = sourceScale.max;
    if constexpr ( isFloat )
    {
        if constexpr ( Norm )
        {
            targetMin = 0;
            targetMax = 1;
        }
        else
        {
            targetMin = vdbVolume.min;
            targetMax = vdbVolume.max;
        }
    }
    else
    {
        targetMin = 0;
        targetMax = std::numeric_limits<T>::max();
    }
    [[maybe_unused]] const float k = ( targetMax - targetMin ) / ( sourceScale.max - sourceScale.min );
    res.min = T( k * ( vdbVolume.min - sourceScale.min ) + targetMin );
    res.max = T( k * ( vdbVolume.max - sourceScale.min ) + targetMin );

    VolumeIndexer indexer( res.dims );
    res.data.resize( indexer.size() );

    if ( !vdbVolume.data )
        return res;

    tbb::enumerable_thread_specific accessorPerThread( vdbVolume.data->getConstAccessor() );
    if ( !ParallelFor( 0_vox, indexer.endId(), [&]( VoxelId i )
    {
        auto& accessor = accessorPerThread.local();
        auto coord = indexer.toPos( i );
        float value = accessor.getValue( openvdb::Coord( coord.x + org.x, coord.y + org.y, coord.z + org.z ) );
        if constexpr ( isFloat && !Norm )
            res.data[i] = T( value );
        else
            res.data[i] = T( std::clamp( ( value - sourceScale.min ) * k + targetMin, targetMin, targetMax ) );
    }, cb ) )
        return unexpectedOperationCanceled();
    return res;
}

Expected<SimpleVolumeMinMax> vdbVolumeToSimpleVolume( const VdbVolume& vdbVolume, const Box3i& activeBox, ProgressCallback cb )
{
    return vdbVolumeToSimpleVolumeImpl<float, false>( vdbVolume, activeBox, {}, cb );
}

Expected<SimpleVolumeMinMax> vdbVolumeToSimpleVolumeNorm( const VdbVolume& vdbVolume, const Box3i& activeBox /*= Box3i()*/,
                                                          std::optional<MinMaxf> sourceScale, ProgressCallback cb /*= {} */ )
{
    return vdbVolumeToSimpleVolumeImpl<float, true>( vdbVolume, activeBox, sourceScale, cb );
}

Expected<SimpleVolumeMinMaxU16> vdbVolumeToSimpleVolumeU16( const VdbVolume& vdbVolume, const Box3i& activeBox,
                                                            std::optional<MinMaxf> sourceScale, ProgressCallback cb )
{
    return vdbVolumeToSimpleVolumeImpl<uint16_t, true>( vdbVolume, activeBox, sourceScale, cb );
}

} //namespace MR
