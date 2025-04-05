#include "MRWeightedPointsShell.h"
#include "MRVoxelsVolume.h"
#include "MRCalcDims.h"
#include "MRMarchingCubes.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRAABBTreePoints.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRIsNaN.h"
#include "MRMesh/MRPointsInBall.h"

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

Expected<Mesh> weightedPointsShell( const PointCloud & cloud, const WeightedPointsShellParametersMetric& params )
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


Expected<Mesh> weightedMeshShell( const Mesh & mesh, const WeightedPointsShellParametersMetric& params )
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

Expected<Mesh> weightedMeshShell( const Mesh& mesh, const WeightedPointsShellParametersRegions& params )
{
    VertBitSet allVerts;
    for ( const auto& reg : params.regions )
        allVerts |= allVerts | reg.verts;

    const auto interRadSq = sqr( params.interpolationDist );
    auto pointWeight = [&params, &mesh, &allVerts, interRadSq] ( VertId v )
    {
        if ( params.regions.empty() )
            return 0.f;
        if ( !allVerts.test( v ) )
            return 0.f;
        float res = 0.f;
        size_t n = 0;

        const auto pt = mesh.points[v];
        findPointsInBall( mesh, Ball3f( pt, interRadSq ), [&n, &res, &params]
            ( const PointsProjectionResult & found, const Vector3f &, Ball3f & )
        {
            auto vv = found.vId;
            for ( const auto& reg : params.regions )
            {
                if ( reg.verts.test( vv ) )
                    res += reg.value;
            }
            n += 1;
            return Processing::Continue;
        } );

        return res / static_cast<float>( n );
    };

    // precalculate the weights
    Vector<float, VertId> weights( allVerts.find_last() + 1 );
    for ( auto v : allVerts )
        weights[v] = pointWeight( v );

    DistanceFromWeightedPointsParams distParams;
    distParams.pointWeight = [weights = std::move( weights ), allVerts] ( VertId v ) mutable
    {
        if ( allVerts.test( v ) )
            return weights[v];
        else
            return 0.f;
    };

    distParams.maxWeight = std::numeric_limits<float>::min();
    for ( auto v : allVerts )
    {
        distParams.maxWeight = std::max( distParams.maxWeight, distParams.pointWeight( v ) );
    }

    WeightedPointsShellParametersMetric resParams{ static_cast<WeightedPointsShellParametersBase>( params ), distParams };

    return weightedMeshShell( mesh, resParams );
}


} //namespace MR
