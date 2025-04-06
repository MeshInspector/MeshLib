#include "MRMeshMath.h"
#include "MRParallelFor.h"
#include "MRTriMath.h"
#include "MRRingIterator.h"
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

namespace
{

/// computes the summed six-fold volume of tetrahedrons with one vertex at (0,0,0) and other three vertices taken from a mesh's triangle
class FaceVolumeCalc
{
public:
    FaceVolumeCalc( const MeshTopology & topology, const VertCoords & points, const FaceBitSet& region) : topology_( topology ), points_( points ), region_( region )
    {}
    FaceVolumeCalc( FaceVolumeCalc& x, tbb::split ) : topology_( x.topology_ ), points_( x.points_ ), region_( x.region_ )
    {}
    void join( const FaceVolumeCalc& y )
    {
        volume_ += y.volume_;
    }

    double volume() const
    {
        return volume_;
    }

    void operator()( const tbb::blocked_range<FaceId>& r )
    {
        for ( FaceId f = r.begin(); f < r.end(); ++f )
        {
            if ( region_.test( f ) && topology_.hasFace( f ) )
            {
                const auto coords = getTriPoints( topology_, points_, f );
                volume_ += mixed( Vector3d( coords[0] ), Vector3d( coords[1] ), Vector3d( coords[2] ) );
            }
        }
    }

private:
    const MeshTopology & topology_;
    const VertCoords & points_;
    const FaceBitSet& region_;
    double volume_{ 0.0 };
};

/// computes the summed six-fold volume of tetrahedrons with one vertex at (0,0,0), another vertex at the center of hole,
/// and other two vertices taken from a hole's edge
class HoleVolumeCalc
{
public:
    HoleVolumeCalc( const MeshTopology & topology, const VertCoords & points, const std::vector<EdgeId>& holeRepresEdges ) : topology_( topology ), points_( points ), holeRepresEdges_( holeRepresEdges )
    {}
    HoleVolumeCalc( HoleVolumeCalc& x, tbb::split ) : topology_( x.topology_ ), points_( x.points_ ), holeRepresEdges_( x.holeRepresEdges_ )
    {}
    void join( const HoleVolumeCalc& y )
    {
        volume_ += y.volume_;
    }

    double volume() const
    {
        return volume_;
    }

    void operator()( const tbb::blocked_range<size_t>& r )
    {
        for ( size_t i = r.begin(); i < r.end(); ++i )
        {
            const auto e0 = holeRepresEdges_[i];
            Vector3d sumBdPos;
            int countBdVerts = 0;
            for ( auto e : leftRing( topology_, e0 ) )
            {
                sumBdPos += Vector3d( orgPnt( topology_, points_, e ) );
                ++countBdVerts;
            }
            Vector3d holeCenter = sumBdPos / double( countBdVerts );
            for ( auto e : leftRing( topology_, e0 ) )
            {
                volume_ += mixed( holeCenter, Vector3d( orgPnt( topology_, points_, e ) ), Vector3d( destPnt( topology_, points_, e ) ) );
            }
        }
    }

private:
    const MeshTopology & topology_;
    const VertCoords & points_;
    const std::vector<EdgeId>& holeRepresEdges_;
    double volume_{ 0.0 };
};

} // anonymous namespace

double volume( const MeshTopology & topology, const VertCoords & points, const FaceBitSet* region /*= nullptr */ )
{
    MR_TIMER
    const auto lastValidFace = topology.lastValidFace();
    const auto& faces = topology.getFaceIds( region );
    FaceVolumeCalc fcalc( topology, points, faces );
    parallel_deterministic_reduce( tbb::blocked_range<FaceId>( 0_f, lastValidFace + 1, 1024 ), fcalc );

    const auto holeRepresEdges = topology.findHoleRepresentiveEdges( region );
    HoleVolumeCalc hcalc( topology, points, holeRepresEdges );
    parallel_deterministic_reduce( tbb::blocked_range<size_t>( size_t( 0 ), holeRepresEdges.size() ), hcalc );

    return ( fcalc.volume() + hcalc.volume() ) / 6.0;
}

double holePerimiter( const MeshTopology & topology, const VertCoords & points, EdgeId e0 )
{
    double res = 0;
    if ( topology.left( e0 ) )
    {
        assert( false );
        return res;
    }

    for ( auto e : leftRing( topology, e0 ) )
    {
        assert( !topology.left( e ) );
        res += edgeLength( topology, points, e );
    }
    return res;
}

Vector3d holeDirArea( const MeshTopology & topology, const VertCoords & points, EdgeId e0 )
{
    Vector3d sum;
    if ( topology.left( e0 ) )
    {
        assert( false );
        return sum;
    }

    Vector3d p0{ orgPnt( topology, points, e0 ) };
    for ( auto e : leftRing0( topology, e0 ) )
    {
        assert( !topology.left( e ) );
        Vector3d p1{ orgPnt( topology, points, e ) };
        Vector3d p2{ destPnt( topology, points, e ) };
        sum += cross( p1 - p0, p2 - p0 );
    }
    return 0.5 * sum;
}

Vector3f leftTangent( const MeshTopology & topology, const VertCoords & points, EdgeId e )
{
    assert( topology.left( e ) );
    const auto lNorm = leftNormal( topology, points, e );
    const auto eDir = edgeVector( topology, points, e ).normalized();
    return cross( lNorm, eDir );
}

Vector3f dirDblArea( const MeshTopology & topology, const VertCoords & points, VertId v )
{
    Vector3f sum;
    for ( EdgeId e : orgRing( topology, v ) )
    {
        if ( topology.left( e ).valid() )
        {
            sum += leftDirDblArea( topology, points, e );
        }
    }
    return sum;
}

Vector3f normal( const MeshTopology & topology, const VertCoords & points, const MeshTriPoint & p )
{
    VertId a, b, c;
    topology.getLeftTriVerts( p.e, a, b, c );
    auto n0 = normal( topology, points, a );
    auto n1 = normal( topology, points, b );
    auto n2 = normal( topology, points, c );
    return p.bary.interpolate( n0, n1, n2 ).normalized();
}

Vector3f pseudonormal( const MeshTopology & topology, const VertCoords & points, VertId v, const FaceBitSet * region )
{
    Vector3f sum;
    for ( EdgeId e : orgRing( topology, v ) )
    {
        const auto l = topology.left( e );
        if ( l && ( !region || region->test( l ) ) )
        {
            auto d0 = edgeVector( topology, points, e );
            auto d1 = edgeVector( topology, points, topology.next( e ) );
            auto angle = MR::angle( d0, d1 );
            auto n = cross( d0, d1 );
            sum += angle * n.normalized();
        }
    }

    return sum.normalized();
}

Vector3f pseudonormal( const MeshTopology & topology, const VertCoords & points, UndirectedEdgeId ue, const FaceBitSet * region )
{
    EdgeId e{ ue };
    auto l = topology.left( e );
    if ( l && region && !region->test( l ) )
        l = {};
    auto r = topology.right( e );
    if ( r && region && !region->test( r ) )
        r = {};
    if ( !l && !r )
        return {};
    if ( !l )
        return normal( topology, points, r );
    if ( !r )
        return normal( topology, points, l );
    auto nl = normal( topology, points, l );
    auto nr = normal( topology, points, r );
    return ( nl + nr ).normalized();
}

Vector3f pseudonormal( const MeshTopology & topology, const VertCoords & points, const MeshTriPoint & p, const FaceBitSet * region )
{
    if ( auto v = p.inVertex( topology ) )
        return pseudonormal( topology, points, v, region );
    if ( auto e = p.onEdge( topology ) )
        return pseudonormal( topology, points, e.e.undirected(), region );
    assert( !region || region->test( topology.left( p.e ) ) );
    return leftNormal( topology, points, p.e );
}

} //namespace MR
