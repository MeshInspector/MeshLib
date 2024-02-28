#include "MRPointsToDistanceVolume.h"
#include "MRSimpleVolume.h"
#include "MRVolumeIndexer.h"
#include "MRParallelFor.h"
#include "MRPointCloud.h"
#include "MRPointsInBall.h"
#include "MRTimer.h"
#include <limits>

namespace MR
{

Expected<SimpleVolume> pointsToDistanceVolume( const PointCloud & cloud, const PointsToDistanceVolumeParams& params )
{
    MR_TIMER
    assert( params.truncationRadius > 0 );
    assert( params.influenceRadius >= params.truncationRadius );
    assert( params.minInfluencePoints >= 1 );
    assert( cloud.hasNormals() );

    SimpleVolume res;
    res.voxelSize = params.voxelSize;
    res.dims = params.dimensions;
    VolumeIndexer indexer( res.dims );
    res.data.resize( indexer.size(), std::numeric_limits<float>::quiet_NaN() );

    if ( !ParallelFor( size_t( 0 ), indexer.size(), [&]( size_t i )
    {
        auto coord = Vector3f( indexer.toPos( VoxelId( i ) ) ) + Vector3f::diagonal( 0.5f );
        auto voxelCenter = params.origin + mult( params.voxelSize, coord );

        float sumDist = 0;
        int num = 0;
        findPointsInBall( cloud, voxelCenter, params.influenceRadius, [&]( VertId v, const Vector3f& p )
        {
            auto tsdf = std::clamp( dot( cloud.normals[v], voxelCenter - p ),
                -params.truncationRadius, params.truncationRadius );
            sumDist += tsdf;
            ++num;
        } );

        if ( num >= params.minInfluencePoints )
            res.data[i] = sumDist / num;
    }, params.cb ) )
        return unexpectedOperationCanceled();

    res.min = -params.truncationRadius;
    res.max =  params.truncationRadius;
    return res;
}

} //namespace MR
