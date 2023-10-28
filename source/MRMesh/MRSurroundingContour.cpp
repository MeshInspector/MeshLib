#include "MRSurroundingContour.h"
#include "MRMesh.h"
#include "MRPlane3.h"
#include "MREdgePaths.h"
#include "MRTimer.h"
#include <array>

namespace MR
{

namespace
{

// return the plane containing points p0 and p1 and parallel to dir,
// with positive half-space to the left from the line p0-p1 if look from dir
Plane3f getPlaneFromTwoPointsAndDir( const Vector3f & p0, const Vector3f & p1, const Vector3f & dir )
{
    return Plane3f::fromDirAndPt( cross( dir, p1 - p0 ).normalized(), 0.5f * ( p0 + p1 ) );
}

void append( std::vector<EdgeId> & to, const std::vector<EdgeId> & with )
{
    to.reserve( to.size() + with.size() );
    for ( auto e : with )
        to.push_back( e );
}

constexpr float PenaltyFactor = 128.0f;

// Given
//   1. a mesh
//   2. a plane
//   3. two mesh edges intersected by the plane, with dest(e0) and org(e1) in the plane's positive part of space
//   4. edge metric
// constructs the path from dest(e0) to org(e1) in positive part of space with minimal summed metric;
// returns the path found appended with e1
static EdgePath smallestPathInPositiveHalfspace(
    const Mesh & mesh, const Plane3f & plane,
    EdgeId e0, EdgeId e1,
    const EdgeMetric & edgeMetric )
{
    assert( plane.distance( mesh.orgPnt( e0 ) ) <= 0 );
    assert( plane.distance( mesh.destPnt( e0 ) ) >= 0 );

    assert( plane.distance( mesh.orgPnt( e1 ) ) >= 0 );
    assert( plane.distance( mesh.destPnt( e1 ) ) <= 0 );

    auto planeMetric = [&]( EdgeId e ) -> float
    {
        const auto o = mesh.orgPnt( e );
        const auto d = mesh.destPnt( e );

        if ( plane.distance( o ) < 0 || plane.distance( d ) < 0 )
            return PenaltyFactor * edgeMetric( e );

        return edgeMetric( e );
    };
    EdgePath res = buildSmallestMetricPathBiDir( mesh.topology, planeMetric, mesh.topology.dest( e0 ), mesh.topology.org( e1 ) );
    res.push_back( e1 );

    assert( isEdgePath( mesh.topology, res ) );
    return res;
}

// Given
//   1. a mesh
//   2. two planes
//   3. two mesh edges intersected by corresponding plane, with org(ei) and dest(ei) in negative and positive parts of space
//   4. edge metric
// constructs the path from dest(e0) to org(e1) in the intersection of plane0' positive half-space and plane1' negative half-space
// with minimal summed metric;
// returns the path found appended with e1
static EdgePath smallestPathInPositiveWedge(
    const Mesh & mesh, const Plane3f & plane0, const Plane3f & plane1,
    EdgeId e0, EdgeId e1,
    const EdgeMetric & edgeMetric )
{
    assert( plane0.distance( mesh.orgPnt( e0 ) ) <= 0 );
    assert( plane0.distance( mesh.destPnt( e0 ) ) >= 0 );

    assert( plane1.distance( mesh.orgPnt( e1 ) ) <= 0 );
    assert( plane1.distance( mesh.destPnt( e1 ) ) >= 0 );

    auto wedgeMetric = [&]( EdgeId e ) -> float
    {
        const auto o = mesh.orgPnt( e );
        const auto d = mesh.destPnt( e );

        if ( plane0.distance( o ) < 0 || plane0.distance( d ) < 0 || plane1.distance( o ) > 0 || plane1.distance( d ) > 0 )
            return PenaltyFactor * edgeMetric( e );

        return edgeMetric( e );
    };
    EdgePath res = buildSmallestMetricPathBiDir( mesh.topology, wedgeMetric, mesh.topology.dest( e0 ), mesh.topology.org( e1 ) );
    res.push_back( e1 );

    assert( isEdgePath( mesh.topology, res ) );
    return res;
}

} //anonymous namespace

EdgeLoop surroundingContour(
    const Mesh & mesh,
    std::vector<EdgeId> includeEdges,
    const EdgeMetric & edgeMetric,
    const Vector3f & dir
)
{
    MR_TIMER
    EdgeLoop res;
    const auto sz = includeEdges.size();
    if ( sz < 2 )
    {
        assert( false );
        return res;
    }
    if ( sz == 2 )
    {
        auto e0 = includeEdges[0];
        auto e1 = includeEdges[1];
        const auto p0 = mesh.edgePoint( e0, 0.5f );
        const auto p1 = mesh.edgePoint( e1, 0.5f );
        const auto midPlane = getPlaneFromTwoPointsAndDir( p0, p1, dir );

        if ( midPlane.distance( mesh.orgPnt( e0 ) ) > 0 )
            e0 = e0.sym();

        if ( midPlane.distance( mesh.orgPnt( e1 ) ) < 0 )
            e1 = e1.sym();

        res = smallestPathInPositiveHalfspace( mesh, midPlane, e0, e1, edgeMetric );
        append( res, smallestPathInPositiveHalfspace( mesh, -midPlane, e1, e0, edgeMetric ) );
    }
    else
    {
        Vector3f sum;
        for ( auto e : includeEdges )
            sum += mesh.edgePoint( e, 0.5f );
        const Vector3f center = sum / float( sz );
        
        const Vector3f dir0 = ( mesh.edgePoint( includeEdges[0], 0.5f ) - center ).normalized();
        const Vector3f dir1 = cross( dir, dir0 ).normalized();
        auto angle = [&]( EdgeId e )
        {
            auto d = mesh.edgePoint( e, 0.5f ) - center;
            return std::atan2( dot( d, dir1 ), dot( d, dir0 ) );
        };
        std::sort( includeEdges.begin(), includeEdges.end(), [&]( EdgeId a, EdgeId b ) { return angle( a ) > angle( b ); } );

        std::vector<Plane3f> planes;
        planes.reserve( sz );
        for ( auto & e : includeEdges )
        {
            auto plane = getPlaneFromTwoPointsAndDir( mesh.edgePoint( e, 0.5f ), center, dir );
            if ( plane.distance( mesh.orgPnt( e ) ) > 0 )
                e = e.sym();
            planes.push_back( std::move( plane ) );
        }

        for ( int i = 0; i + 1 < sz; ++i )
            append( res, smallestPathInPositiveWedge( mesh, planes[i], planes[i+1], includeEdges[i], includeEdges[i+1], edgeMetric ) );
        append( res, smallestPathInPositiveWedge( mesh, planes.back(), planes.front(), includeEdges.back(), includeEdges.front(), edgeMetric ) );
    }
    assert( isEdgeLoop( mesh.topology, res ) );
    return res;
}

} //namespace MR
