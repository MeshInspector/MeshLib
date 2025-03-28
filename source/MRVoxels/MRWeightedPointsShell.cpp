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

    assert( !params.signDistanceByNormal || cloud.hasNormals() );

    return FunctionVolume
    {
        .data = [params, &tree = cloud.getAABBTree(), &cloud] ( const Vector3i& pos ) -> float
        {
            const auto coord = Vector3f( pos ) + Vector3f::diagonal( 0.5f );
            const auto voxelCenter = params.vol.origin + mult( params.vol.voxelSize, coord );
            auto pd = findClosestWeightedPoint( voxelCenter, tree, params.dist );
            if ( !( pd.dist >= params.dist.minDistance && pd.dist < params.dist.maxDistance ) )
                return cQuietNan;
            if ( params.signDistanceByNormal )
            {
                assert( pd.dist >= 0 );
                if ( dot( cloud.normals[pd.vId], voxelCenter - cloud.points[pd.vId] ) < 0 )
                    pd.dist = -pd.dist;
            }
            return pd.dist;
        },
        .dims = params.vol.dimensions,
        .voxelSize = params.vol.voxelSize
    };
}

FunctionVolume weightedMeshToDistanceFunctionVolume( const Mesh & mesh, const WeightedPointsToDistanceVolumeParams& params )
{
    MR_TIMER

    return FunctionVolume
    {
        .data = [params, &mesh] ( const Vector3i& pos ) -> float
        {
            const auto coord = Vector3f( pos ) + Vector3f::diagonal( 0.5f );
            const auto voxelCenter = params.vol.origin + mult( params.vol.voxelSize, coord );
            auto pd = findClosestWeightedMeshPoint( voxelCenter, mesh, params.dist );
            if ( !( pd.dist >= params.dist.minDistance && pd.dist < params.dist.maxDistance ) )
                return cQuietNan;
            if ( params.signDistanceByNormal )
            {
                assert( pd.dist >= 0 );
                if ( dot( mesh.pseudonormal( pd.mtp ), voxelCenter - mesh.triPoint( pd.mtp ) ) < 0 )
                    pd.dist = -pd.dist;
            }
            return pd.dist;
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

    auto distanceOffset = params.signDistanceByNormal ?
        std::abs( params.offset ) : params.offset;

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
            distanceOffset - 1.001f * params.voxelSize, //minDistance
            distanceOffset + 1.001f * params.voxelSize  //maxDistance
        },
        .signDistanceByNormal = params.signDistanceByNormal
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

Expected<Mesh> weightedMeshShell( const Mesh & mesh, const WeightedPointsShellParameters& params )
{
    MR_TIMER

    const auto box = mesh.getBoundingBox().expanded( Vector3f::diagonal( params.offset + params.dist.maxWeight ) );
    const auto [origin, dimensions] = calcOriginAndDimensions( box, params.voxelSize );

    auto distanceOffset = params.signDistanceByNormal ?
        std::abs( params.offset ) : params.offset;

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
            distanceOffset - 1.001f * params.voxelSize, //minDistance
            distanceOffset + 1.001f * params.voxelSize  //maxDistance
        },
        .signDistanceByNormal = params.signDistanceByNormal
    };

    MarchingCubesParams vmParams
    {
        .origin = origin,
        .cb = params.progress,
        .iso = params.offset,
        .lessInside = true
    };

    return marchingCubes( weightedMeshToDistanceFunctionVolume( mesh, wp2vparams ), vmParams );
}

} //namespace MR
