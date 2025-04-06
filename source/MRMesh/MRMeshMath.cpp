#include "MRMeshMath.h"
#include "MRParallelFor.h"
#include "MRTriMath.h"
#include "MRTimer.h"

namespace MR
{

void getLeftTriPoints( const MeshTopology & topology, const VertCoords & points, EdgeId e, Vector3f & v0, Vector3f & v1, Vector3f & v2 )
{
    VertId a, b, c;

    topology.getLeftTriVerts( e, a, b, c );
    v0 = points[a];
    v1 = points[b];
    v2 = points[c];
}

Vector3f triPoint( const MeshTopology & topology, const VertCoords & points, const MeshTriPoint & p )
{
    if ( p.bary.b == 0 )
    {
        // do not require to have triangular face to the left of p.e
        const Vector3f v0 = orgPnt( topology, points, p.e );
        const Vector3f v1 = destPnt( topology, points, p.e );
        return ( 1 - p.bary.a ) * v0 + p.bary.a * v1;
    }
    Vector3f v0, v1, v2;
    getLeftTriPoints( topology, points, p.e, v0, v1, v2 );
    return p.bary.interpolate( v0, v1, v2 );
}

MeshTriPoint toTriPoint( const MeshTopology & topology, const VertCoords & points, FaceId f, const Vector3f & p )
{
    auto e = topology.edgeWithLeft( f );
    Vector3f v0, v1, v2;
    getLeftTriPoints( topology, points, e, v0, v1, v2 );
    return MeshTriPoint( e, p, v0, v1, v2 );
}

MeshTriPoint toTriPoint( const MeshTopology & topology, const VertCoords & points, const PointOnFace & p )
{
    return toTriPoint( topology, points, p.face, p.point );
}

MeshEdgePoint toEdgePoint( const MeshTopology & topology, const VertCoords & points, EdgeId e, const Vector3f & p )
{
    const auto & po = points[ topology.org( e ) ];
    const auto & pd = points[ topology.dest( e ) ];
    const auto dt = dot( p - po , pd - po );
    const auto edgeLenSq = ( pd - po ).lengthSq();
    if ( dt <= 0 || edgeLenSq <= 0 )
        return { e, 0 };
    if ( dt >= edgeLenSq )
        return { e, 1 };
    return { e, dt / edgeLenSq };
}

VertId getClosestVertex( const MeshTopology & topology, const VertCoords & points, const PointOnFace & p )
{
    VertId res, b, c;
    topology.getTriVerts( p.face, res, b, c );
    float closestDistSq = ( points[res] - p.point ).lengthSq();
    if ( auto bDistSq = ( points[b] - p.point ).lengthSq(); bDistSq < closestDistSq )
    {
        res = b;
        closestDistSq = bDistSq;
    }
    if ( auto cDistSq = ( points[c] - p.point ).lengthSq(); cDistSq < closestDistSq )
    {
        res = c;
        closestDistSq = cDistSq;
    }
    return res;
}

UndirectedEdgeId getClosestEdge( const MeshTopology & topology, const VertCoords & points, const PointOnFace & p )
{
    EdgeId e = topology.edgeWithLeft( p.face );
    Vector3f a, b, c;
    getLeftTriPoints( topology, points, e, a, b, c );

    auto distSq = [&]( const LineSegm3f & l )
    {
        return ( p.point - closestPointOnLineSegm( p.point, l ) ).lengthSq();
    };

    UndirectedEdgeId res = e.undirected();
    float closestDistSq = distSq( { a, b } );

    e = topology.prev( e.sym() );
    if ( auto eDistSq = distSq( { b, c } ); eDistSq < closestDistSq )
    {
        res = e.undirected();
        closestDistSq = eDistSq;
    }

    e = topology.prev( e.sym() );
    if ( auto eDistSq = distSq( { c, a } ); eDistSq < closestDistSq )
    {
        res = e.undirected();
        closestDistSq = eDistSq;
    }

    return res;
}

Vector3f triCenter( const MeshTopology & topology, const VertCoords & points, FaceId f )
{
    Vector3f v0, v1, v2;
    getTriPoints( topology, points, f, v0, v1, v2 );
    return ( 1 / 3.0f ) * ( v0 + v1 + v2 );
}

UndirectedEdgeScalars edgeLengths( const MeshTopology & topology, const VertCoords & points )
{
    MR_TIMER
    UndirectedEdgeScalars res( topology.undirectedEdgeSize() );
    ParallelFor( res, [&]( UndirectedEdgeId ue )
    {
        if ( topology.isLoneEdge( ue ) )
            return;
        res[ue] = edgeLength( topology, points, ue );
    } );

    return res;
}

Vector3f leftDirDblArea( const MeshTopology & topology, const VertCoords & points, EdgeId e )
{
    VertId a, b, c;
    topology.getLeftTriVerts( e, a, b, c );
    assert( a.valid() && b.valid() && c.valid() );
    const auto & ap = points[a];
    const auto & bp = points[b];
    const auto & cp = points[c];
    return cross( bp - ap, cp - ap );
}

float triangleAspectRatio( const MeshTopology & topology, const VertCoords & points, FaceId f )
{
    VertId a, b, c;
    topology.getTriVerts( f, a, b, c );
    assert( a.valid() && b.valid() && c.valid() );
    const auto & ap = points[a];
    const auto & bp = points[b];
    const auto & cp = points[c];
    return MR::triangleAspectRatio( ap, bp, cp );
}

float circumcircleDiameterSq( const MeshTopology & topology, const VertCoords & points, FaceId f )
{
    VertId a, b, c;
    topology.getTriVerts( f, a, b, c );
    assert( a.valid() && b.valid() && c.valid() );
    const auto & ap = points[a];
    const auto & bp = points[b];
    const auto & cp = points[c];
    return MR::circumcircleDiameterSq( ap, bp, cp );
}

float circumcircleDiameter( const MeshTopology & topology, const VertCoords & points, FaceId f )
{
    return std::sqrt( circumcircleDiameterSq( topology, points, f ) );
}

double area( const MeshTopology & topology, const VertCoords & points, const FaceBitSet & fs )
{
    MR_TIMER

    return 0.5 * parallel_deterministic_reduce( tbb::blocked_range( 0_f, FaceId{ topology.faceSize() }, 1024 ), 0.0,
    [&] ( const auto & range, double curr )
    {
        for ( FaceId f = range.begin(); f < range.end(); ++f )
            if ( fs.test( f ) && topology.hasFace( f ) )
                curr += dblArea( topology, points, f );
        return curr;
    },
    [] ( auto a, auto b ) { return a + b; } );
}

double projArea( const MeshTopology & topology, const VertCoords & points, const Vector3f & dir, const FaceBitSet & fs )
{
    MR_TIMER

    return 0.5 * parallel_deterministic_reduce( tbb::blocked_range( 0_f, FaceId{ topology.faceSize() }, 1024 ), 0.0,
    [&] ( const auto & range, double curr )
    {
        for ( FaceId f = range.begin(); f < range.end(); ++f )
            if ( fs.test( f ) && topology.hasFace( f ) )
                curr += std::abs( dot( dirDblArea( topology, points, f ), dir ) );
        return curr;
    },
    [] ( auto a, auto b ) { return a + b; } );
}

Vector3d dirArea( const MeshTopology & topology, const VertCoords & points, const FaceBitSet & fs )
{
    MR_TIMER

    return 0.5 * parallel_deterministic_reduce( tbb::blocked_range( 0_f, FaceId{ topology.faceSize() }, 1024 ), Vector3d{},
    [&] ( const auto & range, Vector3d curr )
    {
        for ( FaceId f = range.begin(); f < range.end(); ++f )
            if ( fs.test( f ) && topology.hasFace( f ) )
                curr += Vector3d( dirDblArea( topology, points, f ) );
        return curr;
    },
    [] ( auto a, auto b ) { return a + b; } );
}

} //namespace MR
