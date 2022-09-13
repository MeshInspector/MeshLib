#include "MRSurroundingContour.h"
#include "MRMesh.h"
#include "MRPlane3.h"
#include "MREdgePaths.h"
#include "MRTimer.h"
#include <array>

namespace MR
{

// return the plane containing points p0 and p1 and parallel to dir,
// with positive half-space to the left from the line p0-p1 if look from dir
static Plane3f getPlaneFromTwoPointsAndDir( const Vector3f & p0, const Vector3f & p1, const Vector3f & dir )
{
    return Plane3f::fromDirAndPt( cross( dir, p1 - p0 ).normalized(), 0.5f * ( p0 + p1 ) );
}

static void append( std::vector<EdgeId> & to, const std::vector<EdgeId> & with )
{
    to.reserve( to.size() + with.size() );
    for ( auto e : with )
        to.push_back( e );
}

// edge path from origin e0 to origin e1 in positive half-space, defined by the plane
// containing points mid(e0) and mid(e1) and parallel to dir,
// with positive half-space to the left from the line mid(e0)-mid(e1) if look from dir;
// if either org(e0) or org(e1) are in negative half-space, then e0 or e1.sym() are included in the path
static EdgePath positivePath(
    const Mesh & mesh,
    EdgeId e0, EdgeId e1,
    const EdgeMetric & edgeMetric,
    const Vector3f & dir
)
{
    const auto p0 = mesh.edgePoint( e0, 0.5f );
    const auto p1 = mesh.edgePoint( e1, 0.5f );
    const auto midPlane = getPlaneFromTwoPointsAndDir( p0, p1, dir );

    VertId start = mesh.topology.org( e0 );
    VertId finish = mesh.topology.org( e1 );

    EdgePath res;
    if ( midPlane.distance( mesh.orgPnt( e0 ) ) < 0 )
    {
        res.push_back( e0 );
        start = mesh.topology.dest( e0 );
    }

    if ( midPlane.distance( mesh.orgPnt( e1 ) ) < 0 )
        finish = mesh.topology.dest( e1 );

    auto planeMetric = [&]( EdgeId e ) -> float
    {
        const auto o = mesh.orgPnt( e );
        const auto d = mesh.destPnt( e );

        constexpr float PenaltyFactor = 128.0f;
        if ( midPlane.distance( o ) * midPlane.distance( d ) <= 0 )
            return PenaltyFactor * edgeMetric( e );

        return edgeMetric( e );
    };
    append( res, buildSmallestMetricPathBiDir( mesh.topology, planeMetric, start, finish ) );

    if ( midPlane.distance( mesh.orgPnt( e1 ) ) < 0 )
        res.push_back( e1.sym() );

    assert( isEdgePath( mesh.topology, res ) );
    return res;
}

std::vector<EdgeId> surroundingContour(
    const Mesh & mesh,
    std::vector<EdgeId> includeEdges,
    const EdgeMetric & edgeMetric,
    const Vector3f & dir
)
{
    MR_TIMER
    assert( includeEdges.size() == 2 || includeEdges.size() == 3 );

    if ( includeEdges.size() == 3 )
    {
        std::vector<Vector3f> p( 3 );
        for ( int i = 0; i < 3; ++i )
            p[i] = mesh.edgePoint( includeEdges[i], 0.5f );

        const auto midPlane01 = getPlaneFromTwoPointsAndDir( p[0], p[1], dir );
        if ( midPlane01.distance( p[2] ) > 0 )
        {
            std::swap( includeEdges[0], includeEdges[1] );
            std::swap( p[0], p[1] );
        }
    }

    std::vector<EdgeId> res;
    for ( int i = 0; i + 1 < includeEdges.size(); ++i )
        append( res, positivePath( mesh, includeEdges[i], includeEdges[i + 1], edgeMetric, dir ) );
    append( res, positivePath( mesh, includeEdges.back(), includeEdges[0], edgeMetric, dir ) );
    assert( isEdgeLoop( mesh.topology, res ) );
    return res;
}

} //namespace MR
