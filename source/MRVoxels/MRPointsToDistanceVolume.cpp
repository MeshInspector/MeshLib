#include "MRPointsToDistanceVolume.h"
#include "MRVoxelsVolume.h"
#include "MRMesh/MRVolumeIndexer.h"
#include "MRMesh/MRParallelFor.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRPointsInBall.h"
#include "MRMesh/MRColor.h"
#include "MRMesh/MRTimer.h"
#include <limits>
#include "MRMesh/MRIsNaN.h"

namespace MR
{

FunctionVolume pointsToDistanceFunctionVolume( const PointCloud & cloud, const PointsToDistanceVolumeParams& params )
{
    assert( params.sigma > 0 );
    assert( params.minWeight > 0 );
    assert( params.ptNormals || cloud.hasNormals() );

    return FunctionVolume
    {
        .data = [&cloud, params, inv2SgSq = -0.5f / sqr( params.sigma ),
            &normals = params.ptNormals ? *params.ptNormals : cloud.normals] ( const Vector3i& pos ) -> float
        {
            auto coord = Vector3f( pos ) + Vector3f::diagonal( 0.5f );
            auto voxelCenter = params.origin + mult( params.voxelSize, coord );

            float sumDist = 0;
            float sumWeight = 0;
            findPointsInBall( cloud, voxelCenter, 3 * params.sigma, [&]( VertId v, const Vector3f& p )
            {
                const auto distSq = ( voxelCenter - p ).lengthSq();
                const auto w = std::exp( distSq * inv2SgSq );
                sumWeight += w;
                sumDist += dot( normals[v], voxelCenter - p ) * w;
            } );

            return sumWeight >= params.minWeight ? sumDist / sumWeight : cQuietNan;
        },
        .dims = params.dimensions,
        .voxelSize = params.voxelSize
    };
}

Expected<SimpleVolume> pointsToDistanceVolume( const PointCloud & cloud, const PointsToDistanceVolumeParams& params )
{
    MR_TIMER
    return functionVolumeToSimpleVolume( pointsToDistanceFunctionVolume( cloud, params ), params.cb );
}

Expected<VertColors> calcAvgColors( const PointCloud & cloud, const VertColors & colors,
    const VertCoords & tgtPoints, const VertBitSet & tgtVerts, float sigma, const ProgressCallback & cb )
{
    MR_TIMER
    assert( sigma > 0 );

    VertColors res;
    res.resizeNoInit( tgtPoints.size() );

    const auto inv2SgSq = -0.5f / sqr( sigma );
    if ( !BitSetParallelFor( tgtVerts, [&]( VertId tv )
    {
        const auto pos = tgtPoints[tv];

        Vector4f sumColors;
        float sumWeight = 0;
        findPointsInBall( cloud, pos, 3 * sigma, [&]( VertId v, const Vector3f& p )
        {
            const auto distSq = ( pos - p ).lengthSq();
            const auto w = std::exp( distSq * inv2SgSq );
            sumWeight += w;
            sumColors += Vector4f( colors[v] ) * w;
        } );
        if ( sumWeight > 0 )
            res[tv] = Color( sumColors / sumWeight );
    }, cb ) )
        return unexpectedOperationCanceled();

    return res;
}

} //namespace MR
