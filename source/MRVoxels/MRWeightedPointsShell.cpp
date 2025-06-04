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
#include "MRMesh/MREdgePaths.h"

namespace MR::WeightedShell
{

namespace
{

DistanceVolumeCreationParams getDistanceFieldParams( const Box3f& bbox, const ParametersMetric& params )
{
    const auto box = bbox.expanded( Vector3f::diagonal( params.offset + params.dist.maxWeight ) );
    const auto [origin, dimensions] = calcOriginAndDimensions( box, params.voxelSize );
    return DistanceVolumeCreationParams
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
            params.offset - params.numLayers * params.voxelSize, //minDistance
            params.offset + params.numLayers * params.voxelSize  //maxDistance
        }
    };
}

MarchingCubesParams getMarchingCubesParams( const Box3f& bbox, const ParametersMetric& params )
{
    const auto box = bbox.expanded( Vector3f::diagonal( params.offset + params.dist.maxWeight ) );
    return MarchingCubesParams
    {
        .origin = calcOriginAndDimensions( box, params.voxelSize ).origin,
        .cb = params.progress,
        .iso = params.offset,
        .lessInside = true
    };
}

template <typename T, typename F>
Expected<Mesh> runShell( const T& meshOrCloud, const ParametersMetric& params, const F& buildDistanceField )
{
    auto bbox = meshOrCloud.getBoundingBox();
    auto dfParams = getDistanceFieldParams( bbox, params );
    auto mcParams = getMarchingCubesParams( bbox, params );
    auto df = buildDistanceField( meshOrCloud, dfParams );
    return marchingCubes( df, mcParams );
}

template <typename F>
Expected<Mesh> runShell( const Mesh& mesh, const ParametersRegions& params, const F& buildDistanceField )
{
    MR_TIMER;
    if ( params.regions.empty() )
        spdlog::warn( "WeightedShell::meshShell called without regions. Consider using MR::offsetMesh which is more efficient for constant offset." );

    DistanceFromWeightedPointsParams distParams;
    distParams.maxWeight = 0.f;
    for ( const auto& reg : params.regions )
        distParams.maxWeight = std::max( distParams.maxWeight, reg.weight );


    if ( params.interpolationDist > 0 )
    {
        VertBitSet regionsUnion;
        for ( const auto& reg : params.regions )
            regionsUnion |= reg.verts;
        const bool regionsCoverMesh = mesh.topology.getValidVerts() == regionsUnion;

        // Maximum gradient magnitude = max weight difference divided by the interpolation distance (provided to improve performance)
        // If regions do not cover all mesh, then we need to consider weight=0 for the verts that are not covered.
        //      Otherwise, we consider the maximum difference between different regions.
        float maxWeightChange = distParams.maxWeight - ( regionsCoverMesh ? params.regions[0].weight : 0 );
        for ( const auto& reg : params.regions )
            maxWeightChange = std::max( maxWeightChange, distParams.maxWeight - reg.weight );
        distParams.maxWeightGrad = 1.01f * ( maxWeightChange / params.interpolationDist ); // add extra 10% for floating-point errors in interpolation
    }

    const auto weights = calculateShellWeightsFromRegions( mesh, params.regions, params.interpolationDist );
    distParams.pointWeight = [&weights] ( VertId v )
    {
        return weights[v];
    };
    ParametersMetric resParams{ static_cast< const ParametersBase& >( params ), std::move( distParams ) };
    resParams.dist.bidirectionalMode = params.bidirectionalMode;

    return runShell( mesh, resParams, buildDistanceField );
}

}


VertScalars calculateShellWeightsFromRegions(
    const Mesh& mesh, const std::vector<ParametersRegions::Region>& regions, float interpolationDist )
{
    MR_TIMER;

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
    VertBitSet allVertsExtended = allVerts;
    dilateRegion( mesh, allVertsExtended, interpolationDist );
    BitSetParallelFor( allVertsExtended, [&weights, &pointWeight] ( VertId v )
    {
        weights[v] = pointWeight( v );
    } );

    return weights;
}

FunctionVolume pointsToDistanceVolume( const PointCloud& cloud, const DistanceVolumeCreationParams& params )
{
    return FunctionVolume
    {
        .data = [params, &tree = cloud.getAABBTree()] ( const Vector3i& pos ) -> float
        {
            const auto coord = Vector3f( pos ) + Vector3f::diagonal( 0.5f );
            const auto voxelCenter = params.vol.origin + mult( params.vol.voxelSize, coord );
            auto pd = findClosestWeightedPoint( voxelCenter, tree, params.dist );
            assert( std::isfinite( pd.dist ) );
            if ( pd.dist >= params.dist.maxBidirDist )
                return cQuietNan;
            if ( params.dist.bidirectionalMode && pd.dist < params.dist.minBidirDist )
                return cQuietNan;
            return pd.dist;
        },
        .dims = params.vol.dimensions,
        .voxelSize = params.vol.voxelSize
    };
}

FunctionVolume meshToDistanceVolume( const Mesh& mesh, const DistanceVolumeCreationParams& params )
{
    return FunctionVolume
    {
        .data = [params, &mesh] ( const Vector3i& pos ) -> float
        {
            const auto coord = Vector3f( pos ) + Vector3f::diagonal( 0.5f );
            const auto voxelCenter = params.vol.origin + mult( params.vol.voxelSize, coord );
            auto pd = findClosestWeightedMeshPoint( voxelCenter, mesh, params.dist );
            const auto bdist = pd.bidirDist();
            assert( std::isfinite( bdist ) );
            if ( bdist >= params.dist.maxBidirDist )
                return cQuietNan;
            if ( params.dist.bidirectionalMode && bdist < params.dist.minBidirDist )
                return cQuietNan;
            return pd.dist();
        },
        .dims = params.vol.dimensions,
        .voxelSize = params.vol.voxelSize
    };
}


Expected<Mesh> pointsShell( const PointCloud& cloud, const ParametersMetric& params )
{
    return runShell( cloud, params, pointsToDistanceVolume );
}

Expected<Mesh> pointsShell( const PointCloud& cloud, const VertScalars& pointWeights, const ParametersMetric& params0 )
{
    auto params = params0;
    params.dist.pointWeight = [&pointWeights]( VertId v ){ return pointWeights[v]; };
    return runShell( cloud, params, pointsToDistanceVolume );
}

Expected<Mesh> meshShell( const Mesh& mesh, const ParametersMetric& params )
{
    return runShell( mesh, params, meshToDistanceVolume );
}

Expected<Mesh> meshShell( const Mesh& mesh, const VertScalars& vertWeights, const ParametersMetric& params0 )
{
    auto params = params0;
    params.dist.pointWeight = [&vertWeights]( VertId v ){ return vertWeights[v]; };
    return runShell( mesh, params, meshToDistanceVolume );
}

Expected<Mesh> meshShell( const Mesh& mesh, const ParametersRegions& params )
{
    return runShell( mesh, params, meshToDistanceVolume );
}

Expected<Mesh> meshShell( const Mesh& mesh, const ParametersRegions& params, meshToDistanceVolumeT volumeBuilder )
{
    return runShell( mesh, params, volumeBuilder );
}


} //namespace MR
