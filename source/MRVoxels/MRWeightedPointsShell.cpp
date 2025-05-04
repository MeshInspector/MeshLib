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
#include "MRMesh/MRBitSetParallelFor.h"

#include "MRPch/MRSpdlog.h"
#include "MRMesh/MRParallelFor.h"

namespace MR
{

FunctionVolume weightedPointsToDistanceFunctionVolume( const PointCloud & cloud, const WeightedPointsToDistanceVolumeParams& params )
{
    MR_TIMER;

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

FunctionVolume weightedMeshToDistanceFunctionVolume( const Mesh & mesh, const WeightedMeshPointsToDistanceVolumeParams& params )
{
    MR_TIMER;

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
    MR_TIMER;

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

Expected<Mesh> weightedPointsShell( const PointCloud & cloud, const VertScalars& pointWeights,
                                    const WeightedPointsShellParametersMetric& params0 )
{
    auto params = params0;
    params.dist.pointWeight = [&pointWeights]( VertId v ){ return pointWeights[v]; };
    return weightedPointsShell( cloud, params );
}

Expected<Mesh> weightedMeshShell( const Mesh & mesh, const WeightedPointsShellParametersMetric& params )
{
    MR_TIMER;

    const auto box = mesh.getBoundingBox().expanded( Vector3f::diagonal( params.offset + params.dist.maxWeight ) );
    const auto [origin, dimensions] = calcOriginAndDimensions( box, params.voxelSize );

    auto distanceOffset = params.signDistanceByNormal ?
        std::abs( params.offset ) : params.offset;

    WeightedMeshPointsToDistanceVolumeParams wp2vparams
    {
        .vol =
        {
            .origin = origin,
            .voxelSize = Vector3f::diagonal( params.voxelSize ),
            .dimensions = dimensions,
        },
        .dist = DistanceFromWeightedMeshPointsComputeParams
        {
            DistanceFromWeightedPointsComputeParams{
                params.dist,
                distanceOffset - 1.001f * params.voxelSize, //minDistance
                distanceOffset + 1.001f * params.voxelSize  //maxDistance
            },
            params.bidirectionalMode
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

Expected<Mesh> weightedMeshShell( const Mesh & mesh, const VertScalars& vertWeights, const WeightedPointsShellParametersMetric& params0 )
{
    auto params = params0;
    params.dist.pointWeight = [&vertWeights]( VertId v ){ return vertWeights[v]; };
    return weightedMeshShell( mesh, params );
}

VertScalars calculateShellWeightsFromRegions(
    const Mesh& mesh, const std::vector<WeightedPointsShellParametersRegions::Region>& regions, float interpolationDist )
{
    MR_TIMER;

    if ( regions.empty() )
        spdlog::warn( "weightedMeshShell called without regions. Consider using MR::offsetMesh which is more efficient for constant offset." );

    VertBitSet allVerts;
    for ( const auto& reg : regions )
        allVerts |= reg.verts;

    const float interRadSq = sqr( interpolationDist );
    auto pointWeight = [&regions, &mesh, &allVerts, interRadSq] ( VertId v )
    {
        if ( regions.empty() )
            return 0.f;
        MinMaxf minmax;
        float res = 0.0f;
        size_t n = 0;

        const auto pt = mesh.points[v];
        findPointsInBall( mesh, Ball3f{ pt, interRadSq }, [&n, &res, &regions, &allVerts,&minmax]
            ( const PointsProjectionResult & found, const Vector3f &, Ball3f & )
        {
            auto vv = found.vId;
            for ( const auto& reg : regions )
            {
                if ( reg.verts.test( vv ) )
                {
                    minmax.include( reg.weight );
                    res += reg.weight;
                    n += 1;
                }
            }
            if ( !allVerts.test( vv ) )
            {
                minmax.include( 0.0f );
                n += 1;
            }
            return Processing::Continue;
        } );

        if ( n == 0 )
            return 0.f;
        return std::clamp( res / float( n ), minmax.min, minmax.max ); // not to exceed limits because of floating point errors
    };

    // precalculate the weights
    VertScalars weights( mesh.topology.getValidVerts().find_last() + 1, 0 );
    BitSetParallelFor( allVerts, [&weights, &pointWeight] ( VertId v )
    {
        weights[v] = pointWeight( v );
    } );

    return weights;
}

Expected<Mesh> weightedMeshShell( const Mesh& mesh, const WeightedPointsShellParametersRegions& params )
{
    MR_TIMER;

    auto weights = calculateShellWeightsFromRegions( mesh, params.regions, params.interpolationDist );

    DistanceFromWeightedPointsParams distParams;
    distParams.maxWeight = 0.f;
    for ( const auto& reg : params.regions )
        distParams.maxWeight = std::max( distParams.maxWeight, reg.weight );
    distParams.pointWeight = [&weights] ( VertId v )
    {
        return weights[v];
    };

    ParallelFor( weights, [&] ( VertId i )
    {
        weights[i] -= distParams.maxWeight;
    } );

    WeightedPointsShellParametersMetric resParams{ static_cast< const WeightedPointsShellParametersBase& >( params ), distParams };
    resParams.dist.maxWeight = 0.0f;
    resParams.offset += distParams.maxWeight;

    return weightedMeshShell( mesh, resParams );
}


} //namespace MR
