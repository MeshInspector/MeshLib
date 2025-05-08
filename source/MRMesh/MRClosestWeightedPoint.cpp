#include "MRClosestWeightedPoint.h"
#include "MRPointsInBall.h"
#include "MRMesh.h"
#include "MRTriMath.h"
#include "MRPlane3.h"
#include "MRClosestPointInTriangle.h"

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
    //assert( w <= params_.maxWeight ); // commented due to possible intepolation errors
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

struct ClosestTriPoint
{
    Vector3f pos;
    TriPointf tp;
    float w = 0;
    bool outside = false;
};

std::optional<ClosestTriPoint> findClosestWeightedTriPoint( const Vector3d& locd, const Mesh& mesh, FaceId f, const VertMetric& pointWeight, bool bidirectionalMode )
{
    auto vs = mesh.topology.getTriVerts( f );

    // compute the closest point in double-precision, because float might be not enough
    Triangle3d ps;
    double ws[3];
    for ( int i = 0; i < 3; ++i )
    {
        ps[i] = Vector3d( mesh.points[vs[i]] );
        ws[i] = pointWeight( vs[i] );
    }

    const bool outside = dot( locd - ps[0], dirDblArea( ps ) ) >= 0;
    // considering unsigned distances, each triangle has two planes where euclidean distance equals interpolated point weight
    const auto maybePlane = ( !bidirectionalMode || outside ) ?
        tangentPlaneToSpheres( ps[0], ps[1], ps[2], ws[0], ws[1], ws[2] ) :
        tangentPlaneToSpheres( ps[1], ps[0], ps[2], ws[1], ws[0], ws[2] );
    if ( !maybePlane )
        return {};

    Triangle3d tanPs;
    for ( int i = 0; i < 3; ++i )
        tanPs[i] = maybePlane->project( ps[i] );

    const auto [projD, baryD] = closestPointInTriangle( locd, tanPs[0], tanPs[1], tanPs[2] );

    return ClosestTriPoint
    {
        .pos = Vector3f( baryD.interpolate( ps[0], ps[1], ps[2] ) ), // not projD, since it is on tangent plane
        .tp = TriPointf( baryD ),
        .w = float( baryD.interpolate( ws[0], ws[1], ws[2] ) ),
        .outside = outside
    };
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
    const Mesh& mesh, const DistanceFromWeightedPointsComputeParams& params )
{
    MeshPointAndDistance res{ .dist = params.maxDistance };
    assert( params.pointWeight );
    if ( !params.pointWeight )
        return res;
    BallRadiusAssessor ballRadiusAssessor( params );
    if ( ballRadiusAssessor.maxSearchRadius() < 0 )
        return res;

    const Vector3d locd( loc );
    findBoxedTrisInBall( mesh, { loc, sqr( ballRadiusAssessor.maxSearchRadius() ) }, [&]( FaceId f, Ball3f & ball )
    {
        auto c = findClosestWeightedTriPoint( locd, mesh, f, params.pointWeight, params.bidirectionalMode );
        if ( !c )
            return Processing::Continue;
        c->outside = dot( mesh.pseudonormal( MeshTriPoint{ mesh.topology.edgeWithLeft( f ), c->tp } ), loc - c->pos ) > 0;

        const auto r = distance( loc, c->pos );
        const auto dist = ( params.bidirectionalMode || c->outside ) ? ( r - c->w ) : ( -r - c->w );
        if ( ( params.bidirectionalMode && dist < res.dist ) || ( !params.bidirectionalMode && std::abs( dist ) < std::abs( res.dist ) ) )
        {
            res.dist = dist;
            res.mtp = MeshTriPoint{ mesh.topology.edgeWithLeft( f ), c->tp };
            if ( dist < params.minDistance )
                return Processing::Stop;
        }
        if ( ballRadiusAssessor.pointFound( r, c->w ) )
            ball.radiusSq = sqr( ballRadiusAssessor.maxSearchRadius() );
        return Processing::Continue;
    } );

    return res;
}

} //namespace MR
