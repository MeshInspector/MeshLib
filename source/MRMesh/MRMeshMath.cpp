#include "MRMeshMath.h"
#include "MRParallelFor.h"
#include "MRBitSetParallelFor.h"
#include "MRTriMath.h"
#include "MRRingIterator.h"
#include "MRComputeBoundingBox.h"
#include "MRQuadraticForm.h"
#include "MRLineSegm.h"
#include "MRPlane3.h"
#include "MRTimer.h"

namespace MR
{

LineSegm3f edgeSegment( const MeshTopology & topology, const VertCoords & points, EdgeId e )
{
    return { orgPnt( topology, points, e ), destPnt( topology, points, e ) };
}

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
    MR_TIMER;
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

Vector<Vector3f, VertId> dirDblAreas( const MeshTopology & topology, const VertCoords & points, const VertBitSet * region )
{
    MR_TIMER;
    const auto & vs = topology.getVertIds( region );
    Vector<Vector3f, VertId> res( vs.find_last() + 1 );
    BitSetParallelFor( vs, [&]( VertId v )
    {
        res[v] = dirDblArea( topology, points, v );
    } );
    return res;
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
    MR_TIMER;

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
    MR_TIMER;

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
    MR_TIMER;

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

    double volume()
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

    double volume()
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

class CreaseEdgesCalc
{
public:
    CreaseEdgesCalc( const MeshTopology & topology, const VertCoords & points, float critCos ) : topology_( topology ), points_( points ), critCos_( critCos )
        { edges_.resize( topology_.undirectedEdgeSize() ); }
    CreaseEdgesCalc( CreaseEdgesCalc & x, tbb::split ) : topology_( x.topology_ ), points_( x.points_ ), critCos_( x.critCos_ )
        { edges_.resize( topology_.undirectedEdgeSize() ); }

    void join( const CreaseEdgesCalc & y ) { edges_ |= y.edges_; }

    UndirectedEdgeBitSet takeEdges() { return std::move( edges_ ); }

    void operator()( const tbb::blocked_range<UndirectedEdgeId> & r )
    {
        for ( UndirectedEdgeId ue = r.begin(); ue < r.end(); ++ue )
        {
            if ( topology_.isLoneEdge( ue ) )
                continue;
            auto dihedralCos = dihedralAngleCos( topology_, points_, ue );
            if ( dihedralCos <= critCos_ )
                edges_.set( ue );
        }
    }

private:
    const MeshTopology & topology_;
    const VertCoords & points_;
    float critCos_ = 1;
    UndirectedEdgeBitSet edges_;
};

class FaceBoundingBoxCalc
{
public:
    FaceBoundingBoxCalc( const MeshTopology & topology, const VertCoords & points, const FaceBitSet& region, const AffineXf3f* toWorld ) : topology_( topology ), points_( points ), region_( region ), toWorld_( toWorld ) {}
    FaceBoundingBoxCalc( FaceBoundingBoxCalc& x, tbb::split ) : topology_( x.topology_ ), points_( x.points_ ), region_( x.region_ ), toWorld_( x.toWorld_ ) {}
    void join( const FaceBoundingBoxCalc & y ) { box_.include( y.box_ ); }

    const Box3f & box() const { return box_; }

    void operator()( const tbb::blocked_range<FaceId> & r )
    {
        for ( FaceId f = r.begin(); f < r.end(); ++f )
        {
            if ( region_.test( f ) && topology_.hasFace( f ) )
            {
                for ( EdgeId e : leftRing( topology_, f ) )
                {
                    box_.include( toWorld_ ? ( *toWorld_ )( points_[topology_.org( e )] ) : points_[topology_.org( e )] );
                }
            }
        }
    }

private:
    const MeshTopology & topology_;
    const VertCoords & points_;
    const FaceBitSet & region_;
    Box3f box_;
    const AffineXf3f* toWorld_ = nullptr;
};

} // anonymous namespace

double volume( const MeshTopology & topology, const VertCoords & points, const FaceBitSet* region /*= nullptr */ )
{
    MR_TIMER;
    const auto lastValidFace = topology.lastValidFace();
    const auto& faces = topology.getFaceIds( region );
    FaceVolumeCalc fcalc( topology, points, faces );
    parallel_deterministic_reduce( tbb::blocked_range<FaceId>( 0_f, lastValidFace + 1, 1024 ), fcalc );

    const auto holeRepresEdges = topology.findHoleRepresentiveEdges( region );
    HoleVolumeCalc hcalc( topology, points, holeRepresEdges );
    parallel_deterministic_reduce( tbb::blocked_range<size_t>( size_t( 0 ), holeRepresEdges.size() ), hcalc );

    return ( fcalc.volume() + hcalc.volume() ) / 6.0;
}

Box3f computeBoundingBox( const MeshTopology & topology, const VertCoords & points, const FaceBitSet * region, const AffineXf3f* toWorld )
{
    if ( !region )
        return computeBoundingBox( points, topology.getValidVerts(), toWorld );

    MR_TIMER;
    const auto lastValidFace = topology.lastValidFace();

    FaceBoundingBoxCalc calc( topology, points, *region, toWorld );
    parallel_reduce( tbb::blocked_range<FaceId>( 0_f, lastValidFace + 1 ), calc );
    return calc.box();
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

Plane3f getPlane3f( const MeshTopology & topology, const VertCoords & points, FaceId f )
{
    VertId a, b, c;
    topology.getTriVerts( f, a, b, c );
    assert( a.valid() && b.valid() && c.valid() );
    const Vector3f ap{ points[a] };
    const Vector3f bp{ points[b] };
    const Vector3f cp{ points[c] };
    return Plane3f::fromDirAndPt( cross( bp - ap, cp - ap ).normalized(), ap );
}

Plane3d getPlane3d( const MeshTopology & topology, const VertCoords & points, FaceId f )
{
    VertId a, b, c;
    topology.getTriVerts( f, a, b, c );
    assert( a.valid() && b.valid() && c.valid() );
    const Vector3d ap{ points[a] };
    const Vector3d bp{ points[b] };
    const Vector3d cp{ points[c] };
    return Plane3d::fromDirAndPt( cross( bp - ap, cp - ap ).normalized(), ap );
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
    if ( p.bary.b == 0 )
    {
        // do not require to have triangular face to the left of p.e
        const Vector3f n0 = normal( topology, points, topology.org( p.e ) );
        const Vector3f n1 = normal( topology, points, topology.dest( p.e ) );
        return lerp( n0, n1, p.bary.a ).normalized();
    }
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

float sumAngles( const MeshTopology & topology, const VertCoords & points, VertId v, bool * outBoundaryVert )
{
    if ( outBoundaryVert )
        *outBoundaryVert = false;
    float sum = 0;
    for ( EdgeId e : orgRing( topology, v ) )
    {
        if ( topology.left( e ).valid() )
        {
            auto d0 = edgeVector( topology, points, e );
            auto d1 = edgeVector( topology, points, topology.next( e ) );
            auto angle = MR::angle( d0, d1 );
            sum += angle;
        }
        else if ( outBoundaryVert )
            *outBoundaryVert = true;
    }
    return sum;
}

Expected<VertBitSet> findSpikeVertices( const MeshTopology & topology, const VertCoords & points, float minSumAngle, const VertBitSet * region, const ProgressCallback& cb )
{
    MR_TIMER;
    const VertBitSet & testVerts = topology.getVertIds( region );
    VertBitSet res( testVerts.size() );
    auto completed = BitSetParallelFor( testVerts, [&]( VertId v )
    {
        bool boundaryVert = false;
        auto a = sumAngles( topology, points, v, &boundaryVert );
        if ( !boundaryVert && a < minSumAngle )
            res.set( v );
    }, cb );

    if ( !completed )
        return unexpectedOperationCanceled();

    return res;
}

float dihedralAngleSin( const MeshTopology & topology, const VertCoords & points, UndirectedEdgeId ue )
{
    EdgeId e{ ue };
    if ( topology.isBdEdge( e ) )
        return 0;
    return MR::dihedralAngleSin( leftNormal( topology, points, e ), leftNormal( topology, points, e.sym() ), edgeVector( topology, points, e ) );
}

float dihedralAngleCos( const MeshTopology & topology, const VertCoords & points, UndirectedEdgeId ue )
{
    EdgeId e{ ue };
    if ( topology.isBdEdge( e ) )
        return 1;
    return MR::dihedralAngleCos( leftNormal( topology, points, e ), leftNormal( topology, points, e.sym() ) );
}

float dihedralAngle( const MeshTopology & topology, const VertCoords & points, UndirectedEdgeId ue )
{
    EdgeId e{ ue };
    if ( topology.isBdEdge( e ) )
        return 0;
    return MR::dihedralAngle( leftNormal( topology, points, e ), leftNormal( topology, points, e.sym() ), edgeVector( topology, points, e ) );
}

float discreteMeanCurvature( const MeshTopology & topology, const VertCoords & points, VertId v )
{
    float sumArea = 0;
    float sumAngLen = 0;
    for ( EdgeId e : orgRing( topology, v ) )
    {
        auto l = topology.left( e );
        if ( !l )
            continue; // area( l ) is not defined and dihedralAngle( e ) = 0
        sumArea += area( topology, points, l );
        sumAngLen += dihedralAngle( topology, points, e.undirected() ) * edgeLength( topology, points, e.undirected() );
    }
    // sumAngLen / (2*2) because of mean curvature definition * each edge has 2 vertices,
    // sumArea / 3 because each triangle has 3 vertices
    return ( sumArea > 0 ) ? 0.75f * sumAngLen / sumArea : 0;
}

float discreteMeanCurvature( const MeshTopology & topology, const VertCoords & points, UndirectedEdgeId ue )
{
    EdgeId e = ue;
    if ( topology.isBdEdge( e ) )
        return 0;
    float sumArea = area( topology, points, topology.left( e ) ) + area( topology, points, topology.right( e ) );
    float sumAngLen = dihedralAngle( topology, points, e.undirected() ) * edgeLength( topology, points, e.undirected() );
    // sumAngLen / 2 because of mean curvature definition,
    // sumArea / 3 because each triangle has 3 edges
    return ( sumArea > 0 ) ? 1.5f * sumAngLen / sumArea : 0;
}

UndirectedEdgeBitSet findCreaseEdges( const MeshTopology & topology, const VertCoords & points, float angleFromPlanar )
{
    MR_TIMER;
    assert( angleFromPlanar > 0 && angleFromPlanar < PI );
    const float critCos = std::cos( angleFromPlanar );
    CreaseEdgesCalc calc( topology, points, critCos );
    parallel_reduce( tbb::blocked_range<UndirectedEdgeId>( 0_ue, UndirectedEdgeId{ topology.undirectedEdgeSize() } ), calc );
    return calc.takeEdges();
}

float leftCotan( const MeshTopology & topology, const VertCoords & points, EdgeId e )
{
    if ( !topology.left( e ).valid() )
        return 0;
    return MR::cotan( getLeftTriPoints( topology, points, e ) );
}

QuadraticForm3f quadraticForm( const MeshTopology & topology, const VertCoords & points, VertId v, bool angleWeigted, const FaceBitSet * region, const UndirectedEdgeBitSet * creases )
{
    QuadraticForm3f qf;
    for ( EdgeId e : orgRing( topology, v ) )
    {
        if ( topology.isBdEdge( e, region ) || ( creases && creases->test( e ) ) )
        {
            // zero-length boundary edge is treated as uniform stabilizer: all shift directions are equally penalized,
            // otherwise it penalizes the shift proportionally to the distance from the line containing the edge
            qf.addDistToLine( edgeVector( topology, points, e ).normalized() );
        }
        if ( topology.left( e ) ) // intentionally do not check that left face is in region to respect its plane as well
        {
            if ( angleWeigted )
            {
                auto d0 = edgeVector( topology, points, e );
                auto d1 = edgeVector( topology, points, topology.next( e ) );
                auto angle = MR::angle( d0, d1 );
                static constexpr float INV_PIF = 1 / PI_F;
                qf.addDistToPlane( leftNormal( topology, points, e ), angle * INV_PIF );
            }
            else
            {
                // zero-area triangle is treated as no triangle with no penalty at all,
                // otherwise it penalizes the shift proportionally to the distance from the plane containing the triangle
                qf.addDistToPlane( leftNormal( topology, points, e ) );
            }
        }
    }
    return qf;
}

float averageEdgeLength( const MeshTopology & topology, const VertCoords & points )
{
    MR_TIMER;
    struct S
    {
        double sum = 0;
        int n = 0;
        S & operator +=( const S & b )
        {
            sum += b.sum;
            n += b.n;
            return *this;
        }
    };
    S s = parallel_deterministic_reduce( tbb::blocked_range( 0_ue, UndirectedEdgeId{ topology.undirectedEdgeSize() }, 1024 ), S{},
        [&] ( const auto & range, S curr )
        {
            for ( UndirectedEdgeId ue = range.begin(); ue < range.end(); ++ue )
                if ( !topology.isLoneEdge( ue ) )
                {
                    curr.sum += edgeLength( topology, points, ue );
                    ++curr.n;
                }
            return curr;
        },
        [] ( S a, const S & b ) { a += b; return a; }
    );

    return s.n > 0 ? float( s.sum / s.n ) : 0.0f;
}

Vector3f findCenterFromPoints( const MeshTopology & topology, const VertCoords & points )
{
    MR_TIMER;
    if ( topology.numValidVerts() <= 0 )
    {
        assert( false );
        return {};
    }
    auto sumPos = parallel_deterministic_reduce( tbb::blocked_range( 0_v, VertId{ topology.vertSize() }, 1024 ), Vector3d{},
    [&] ( const auto & range, Vector3d curr )
    {
        for ( VertId v = range.begin(); v < range.end(); ++v )
            if ( topology.hasVert( v ) )
                curr += Vector3d{ points[v] };
        return curr;
    },
    [] ( auto a, auto b ) { return a + b; } );
    return Vector3f{ sumPos / (double)topology.numValidVerts() };
}

Vector3f findCenterFromFaces( const MeshTopology & topology, const VertCoords & points )
{
    MR_TIMER;
    struct Acc
    {
        Vector3d areaPos;
        double area = 0;
        Acc operator +( const Acc & b )
        {
            return {
                .areaPos = areaPos + b.areaPos,
                .area = area + b.area
            };
        }
    };
    auto acc = parallel_deterministic_reduce( tbb::blocked_range( 0_f, FaceId{ topology.faceSize() }, 1024 ), Acc{},
    [&] ( const auto & range, Acc curr )
    {
        for ( FaceId f = range.begin(); f < range.end(); ++f )
            if ( topology.hasFace( f ) )
            {
                double triArea = area( topology, points, f );
                Vector3d center( triCenter( topology, points, f ) );
                curr.area += triArea;
                curr.areaPos += center * triArea;
            }
        return curr;
    },
    [] ( auto a, auto b ) { return a + b; } );
    if ( acc.area <= 0 )
    {
        assert( false );
        return {};
    }
    return Vector3f{ acc.areaPos / acc.area };
}

Vector3f findCenterFromBBox( const MeshTopology & topology, const VertCoords & points )
{
    return computeBoundingBox( points, topology.getValidVerts() ).center();
}

} //namespace MR
