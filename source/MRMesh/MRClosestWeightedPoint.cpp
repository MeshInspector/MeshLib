#include "MRClosestWeightedPoint.h"
#include "MRPointsInBall.h"
#include "MRMesh.h"

namespace MR
{

PointAndDistance findClosestWeightedPoint( const Vector3f & loc,
    const AABBTreePoints& tree, const DistanceFromWeightedPointsComputeParams& params )
{
    assert( params.minDistance <= params.maxDistance );
    assert( params.maxDistance >= 0 );
    assert( params.maxWeightGrad >= 0 );
    // if params.maxWeightGrad == 0 then you need to find euclidean closest point - a much simpler algorithm than below

    PointAndDistance res{ .dist = params.maxDistance };
    assert( params.pointWeight );
    if ( !params.pointWeight )
        return res;
    auto maxSearchRadius = params.maxDistance + params.maxWeight;
    if ( maxSearchRadius < 0 )
        return res;

    /// the weight of hypothetical point at loc is no more than this value
    auto maxLocWeight = params.maxWeight;

    findPointsInBall( tree, { loc, sqr( maxSearchRadius ) }, [&]( const PointsProjectionResult & found, const Vector3f &, Ball3f & ball )
    {
        const auto r = std::sqrt( found.distSq );
        const auto w = params.pointWeight( found.vId );
        assert( w <= params.maxWeight );
        assert( w <= maxLocWeight + r * params.maxWeightGrad );
        auto dist = r - w;
        if ( dist < res.dist )
        {
            res.dist = dist;
            res.vId = found.vId;
            if ( dist < params.minDistance )
                return Processing::Stop;
        }
        if ( params.maxWeightGrad < 1 )
        {
            // assess the maximal weight of hypothetical point at loc
            const auto locWeight = w + r * params.maxWeightGrad;
            if ( locWeight < maxLocWeight )
            {
                // try to reduce search radius knowing that the weights of nearby points are limited by known gradient
                maxLocWeight = locWeight;
                const auto searchRadius = ( params.maxDistance + maxLocWeight ) / ( 1 - params.maxWeightGrad );
                if ( searchRadius < maxSearchRadius )
                {
                    maxSearchRadius = searchRadius;
                    ball.radiusSq = sqr( maxSearchRadius );
                }
            }
        }
        return Processing::Continue;
    } );
    return res;
}

MeshPointAndDistance findClosestWeightedMeshPoint( const Vector3f& loc,
    const MeshPart& mp, const DistanceFromWeightedPointsComputeParams& params )
{
    MeshPointAndDistance res;
    // first consider only mesh vertices ignoring triangles
    {
        auto ptRes = findClosestWeightedPoint( loc, mp.mesh.getAABBTreePoints(), params );
        res.dist = ptRes.dist;
        if ( ptRes.vId )
            res.mtp = MeshTriPoint( mp.mesh.topology, ptRes.vId );
    }

    return res;
}

} //namespace MR
