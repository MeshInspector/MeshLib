#include "MRWeightedPointsShell.h"
#include "MRVoxelsVolume.h"
#include "MRCalcDims.h"
#include "MRMarchingCubes.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRAABBTreePoints.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRIsNaN.h"

namespace MR
{

FunctionVolume weightedPointsToDistanceFunctionVolume( const PointCloud & cloud, const WeightedPointsToDistanceVolumeParams& params )
{
    MR_TIMER

    return FunctionVolume
    {
        .data = [params, &tree = cloud.getAABBTree()] ( const Vector3i& pos ) -> float
        {
            const auto coord = Vector3f( pos ) + Vector3f::diagonal( 0.5f );
            const auto voxelCenter = params.vol.origin + mult( params.vol.voxelSize, coord );
            auto pd = findClosestWeightedPoint( voxelCenter, tree, params.dist );
            return ( pd.dist >= params.dist.minDistance && pd.dist < params.dist.maxDistance ) ? pd.dist : cQuietNan;
        },
        .dims = params.vol.dimensions,
        .voxelSize = params.vol.voxelSize
    };
}

Expected<Mesh> weightedPointsShell( const PointCloud & cloud, const WeightedPointsShellParameters& params )
{
    MR_TIMER

    const auto box = cloud.getBoundingBox().expanded( Vector3f::diagonal( params.offset + params.dist.maxWeight ) );
    const auto [origin, dimensions] = calcOriginAndDimensions( box, params.voxelSize );

    WeightedPointsToDistanceVolumeParams wp2vparams
    {
        .vol =
        {
            .origin = origin,
            .voxelSize = Vector3f::diagonal( params.voxelSize ),
            .dimensions = dimensions,
        },
        .dist =
        {
            params.dist,
            params.offset - 1.001f * params.voxelSize, //minDistance
            params.offset + 1.001f * params.voxelSize  //maxDistance
        }
    };

    MarchingCubesParams vmParams
    {
        .origin = origin,
        .cb = params.progress,
        .iso = params.offset,
        .lessInside = true
    };

    return marchingCubes( weightedPointsToDistanceFunctionVolume( cloud, wp2vparams ), vmParams );
}

} //namespace MR
