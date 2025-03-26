#include "MRClosestWeightedPoint.h"
#include "MRPointsInBall.h"
#include "MRMesh.h"

namespace MR
{

namespace
{

class BallRadiusAssessor
{
public:
    explicit BallRadiusAssessor( const DistanceFromWeightedPointsComputeParams& params )
        : params_( params )
        , maxSearchRadius_( params.maxDistance + params.maxWeight )
        , maxLocWeight_ ( params.maxWeight )
    {
    }

    /// call this function on each found point:
    /// \param r distance from requested location to found point
    /// \param w weight of found point
    /// \return true if maxSearchRadius was reduced during the call
    bool pointFound( float r, float w );

    float maxSearchRadius() const { return maxSearchRadius_; }

private:
    const DistanceFromWeightedPointsComputeParams& params_;

    float maxSearchRadius_ = 0;

    /// the weight of hypothetical point at loc is no more than this value
    float maxLocWeight_ = 0;
};

bool BallRadiusAssessor::pointFound( float r, float w )
{
    assert( r >= 0 );
    assert( w <= params_.maxWeight );
    assert( w <= maxLocWeight_ + r * params_.maxWeightGrad );
    if ( params_.maxWeightGrad < 1 )
    {
        // assess the maximal weight of hypothetical point at loc
        const auto locWeight = w + r * params_.maxWeightGrad;
        if ( locWeight < maxLocWeight_ )
        {
            // try to reduce search radius knowing that the weights of nearby points are limited by known gradient
            maxLocWeight_ = locWeight;
            const auto searchRadius = ( params_.maxDistance + maxLocWeight_ ) / ( 1 - params_.maxWeightGrad );
            assert( searchRadius >= 0 ); // if params_.maxDistance + maxLocWeight_ >= 0 and params_.maxWeightGrad < 1
            if ( searchRadius < maxSearchRadius_ )
            {
                maxSearchRadius_ = searchRadius;
                return true;
            }
        }
    }
    return false;
}

} // anonymous namespace

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
    BallRadiusAssessor ballRadiusAssessor( params );
    if ( ballRadiusAssessor.maxSearchRadius() < 0 )
        return res;

    findPointsInBall( tree, { loc, sqr( ballRadiusAssessor.maxSearchRadius() ) }, [&]( const PointsProjectionResult & found, const Vector3f &, Ball3f & ball )
    {
        const auto r = std::sqrt( found.distSq );
        const auto w = params.pointWeight( found.vId );
        auto dist = r - w;
        if ( dist < res.dist )
        {
            res.dist = dist;
            res.vId = found.vId;
            if ( dist < params.minDistance )
                return Processing::Stop;
        }
        if ( ballRadiusAssessor.pointFound( r, w ) )
            ball.radiusSq = sqr( ballRadiusAssessor.maxSearchRadius() );
        return Processing::Continue;
    } );
    return res;
}

MeshPointAndDistance findClosestWeightedMeshPoint( const Vector3f& loc,
    const MeshPart& mp, const DistanceFromWeightedPointsComputeParams& params )
{
    MeshPointAndDistance res;
    assert( params.pointWeight );
    if ( !params.pointWeight )
        return res;
    BallRadiusAssessor ballRadiusAssessor( params );
    if ( ballRadiusAssessor.maxSearchRadius() < 0 )
        return res;

    // first consider only mesh vertices ignoring triangles
    {
        auto ptRes = findClosestWeightedPoint( loc, mp.mesh.getAABBTreePoints(), params );
        res.dist = ptRes.dist;
        if ( ptRes.vId )
        {
            res.mtp = MeshTriPoint( mp.mesh.topology, ptRes.vId );
            const auto r = distance( loc, mp.mesh.points[ptRes.vId] );
            const auto w = params.pointWeight( ptRes.vId );
            ballRadiusAssessor.pointFound( r, w );
        }
    }

    return res;
}

} //namespace MR
