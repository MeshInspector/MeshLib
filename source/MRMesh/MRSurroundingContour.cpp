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
//   3. two mesh edges, each intersected by corresponding plane, with org(ei) and dest(ei) in negative and positive parts of space
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

// Given
//   1. a mesh
//   2. a plane
//   3. two mesh vertices located on the plane
//   4. edge metric
// constructs the path from v0 to v1 in positive part of space with minimal summed metric;
// returns the path found
static EdgePath smallestPathInPositiveHalfspace(
    const Mesh & mesh, const Plane3f & plane,
    VertId v0, VertId v1,
    const EdgeMetric & edgeMetric )
{
    auto planeMetric = [&]( EdgeId e ) -> float
    {
        const auto ov = mesh.topology.org( e );
        if ( ov != v0 && ov != v1 )
        {
            const auto o = mesh.points[ov];
            if ( plane.distance( o ) < 0 )
                return PenaltyFactor * edgeMetric( e );
        }
        const auto dv = mesh.topology.dest( e );
        if ( dv != v0 && dv != v1 )
        {
            const auto d = mesh.points[dv];
            if ( plane.distance( d ) < 0 )
                return PenaltyFactor * edgeMetric( e );
        }
        return edgeMetric( e );
    };
    return buildSmallestMetricPathBiDir( mesh.topology, planeMetric, v0, v1 );
}

// Given
//   1. a mesh
//   2. two planes
//   3. two mesh vertices, each located on corresponding plane
//   4. edge metric
// constructs the path from v0 to v1 in the intersection of plane0' positive half-space and plane1' negative half-space
// with minimal summed metric;
// returns the path found
static EdgePath smallestPathInPositiveWedge(
    const Mesh & mesh, const Plane3f & plane0, const Plane3f & plane1,
    VertId v0, VertId v1,
    const EdgeMetric & edgeMetric )
{
    auto wedgeMetric = [&]( EdgeId e ) -> float
    {
        const auto ov = mesh.topology.org( e );
        if ( ov != v0 && ov != v1 )
        {
            const auto o = mesh.points[ov];
            if ( plane0.distance( o ) < 0 || plane1.distance( o ) > 0 )
                return PenaltyFactor * edgeMetric( e );
        }
        const auto dv = mesh.topology.dest( e );
        if ( dv != v0 && dv != v1 )
        {
            const auto d = mesh.points[dv];
            if ( plane0.distance( d ) < 0 || plane1.distance( d ) > 0 )
                return PenaltyFactor * edgeMetric( e );
        }
        return edgeMetric( e );
    };
    return buildSmallestMetricPathBiDir( mesh.topology, wedgeMetric, v0, v1 );
}

} //anonymous namespace

Expected<EdgeLoop> surroundingContour(
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
        return unexpected( "Too few key edges" );
    if ( sz == 2 )
    {
        if ( includeEdges[0].undirected() == includeEdges[1].undirected() )
            return unexpected( "Two key points are the same" );
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
        // remove repeating edges
        includeEdges.erase( std::unique( includeEdges.begin(), includeEdges.end(), []( UndirectedEdgeId a, UndirectedEdgeId b ) { return a == b; } ), includeEdges.end() );
        while ( includeEdges.size() > 1 && includeEdges.front().undirected() == includeEdges.back().undirected() )
            includeEdges.pop_back();
        if ( includeEdges.size() < 2 )
            return unexpected( "Too few key edges after removing duplicates" );

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
    if ( !isEdgeLoop( mesh.topology, res ) )
        return unexpected( "Key edges are located on different connected components" );
    return res;
}

Expected<EdgeLoop> surroundingContour(
    const Mesh & mesh,
    std::vector<VertId> keyVertices,
    const EdgeMetric & edgeMetric,
    const Vector3f & dir
)
{
    MR_TIMER
    EdgeLoop res;
    const auto sz = keyVertices.size();
    if ( sz < 2 )
        return unexpected( "Too few key vertices" );

    if ( sz == 2 )
    {
        if ( keyVertices[0] == keyVertices[1] )
            return unexpected( "Two key points are the same" );
        auto v0 = keyVertices[0];
        auto v1 = keyVertices[1];
        const auto p0 = mesh.points[v0];
        const auto p1 = mesh.points[v1];
        const auto midPlane = getPlaneFromTwoPointsAndDir( p0, p1, dir );

        res = smallestPathInPositiveHalfspace( mesh, midPlane, v0, v1, edgeMetric );
        append( res, smallestPathInPositiveHalfspace( mesh, -midPlane, v1, v0, edgeMetric ) );
    }
    else
    {
        Vector3f sum;
        for ( auto v : keyVertices )
            sum += mesh.points[v];
        const Vector3f center = sum / float( sz );
        
        const Vector3f dir0 = ( mesh.points[keyVertices[0]] - center ).normalized();
        const Vector3f dir1 = cross( dir, dir0 ).normalized();
        auto angle = [&]( VertId v )
        {
            auto d = mesh.points[v] - center;
            return std::atan2( dot( d, dir1 ), dot( d, dir0 ) );
        };
        std::sort( keyVertices.begin(), keyVertices.end(), [&]( VertId a, VertId b ) { return angle( a ) > angle( b ); } );
        // remove repeating vertices
        keyVertices.erase( std::unique( keyVertices.begin(), keyVertices.end() ), keyVertices.end() );
        while ( keyVertices.size() > 1 && keyVertices.front() == keyVertices.back() )
            keyVertices.pop_back();
        if ( keyVertices.size() < 2 )
            return unexpected( "Too few key vertices after removing duplicates" );

        std::vector<Plane3f> planes;
        planes.reserve( sz );
        for ( auto v : keyVertices )
            planes.push_back( getPlaneFromTwoPointsAndDir( mesh.points[v], center, dir ) );

        for ( int i = 0; i + 1 < sz; ++i )
            append( res, smallestPathInPositiveWedge( mesh, planes[i], planes[i+1], keyVertices[i], keyVertices[i+1], edgeMetric ) );
        append( res, smallestPathInPositiveWedge( mesh, planes.back(), planes.front(), keyVertices.back(), keyVertices.front(), edgeMetric ) );
    }
    if ( !isEdgeLoop( mesh.topology, res ) )
        return unexpected( "Key vertices are located on different connected components" );
    return res;
}

} //namespace MR
