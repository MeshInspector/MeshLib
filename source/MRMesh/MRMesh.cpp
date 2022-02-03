#include "MRMesh.h"
#include "MRBox.h"
#include "MRAffineXf3.h"
#include "MRBitSet.h"
#include "MRTimer.h"
#include "MRRingIterator.h"
#include "MRMeshTriPoint.h"
#include "MRBitSetParallelFor.h"
#include "MRAABBTree.h"
#include "MRTriangleIntersection.h"
#include "MRMeshIntersect.h"
#include "MRLine3.h"
#include "MRLineSegm.h"
#include "MRConstants.h"
#include "MRPch/MRTBB.h"

namespace
{
using namespace MR;

// makes edge loop from points, all connected but not first with last edge
std::vector<EdgeId> sMakeDisclosedEdgeLoop( Mesh& mesh, const std::vector<Vector3f>& contourPoints )
{
    std::vector<EdgeId> newEdges( contourPoints.size() );
    for ( int i = 0; i < contourPoints.size(); ++i )
    {
        auto newVert = mesh.addPoint( contourPoints[i] );
        newEdges[i] = mesh.topology.makeEdge();
        mesh.topology.setOrg( newEdges[i], newVert );
    }
    for ( int i = 0; i + 1 < newEdges.size(); ++i )
    {
        mesh.topology.splice( newEdges[( i + 1 ) % newEdges.size()], newEdges[i].sym() );
    }
    return newEdges;
}
}

namespace MR
{

void Mesh::getLeftTriPoints( EdgeId e, Vector3f & v0, Vector3f & v1, Vector3f & v2 ) const
{
    VertId a, b, c;
    topology.getLeftTriVerts( e, a, b, c );
    v0 = points[a];
    v1 = points[b];
    v2 = points[c];
}

Vector3f Mesh::triPoint( const MeshTriPoint & p ) const
{
    if ( p.bary.b == 0 )
    {
        // do not require to have triangular face to the left of p.e
        const Vector3f v0 = orgPnt( p.e );
        const Vector3f v1 = destPnt( p.e );
        return ( 1 - p.bary.a ) * v0 + p.bary.a * v1;
    }
    Vector3f v0, v1, v2;
    getLeftTriPoints( p.e, v0, v1, v2 );
    return p.bary.interpolate( v0, v1, v2 );
}

MeshTriPoint Mesh::toTriPoint( FaceId f, const Vector3f & p ) const
{
    auto e = topology.edgeWithLeft( f );
    Vector3f v0, v1, v2;
    getLeftTriPoints( e, v0, v1, v2 );
    return MeshTriPoint( e, p, v0, v1, v2 );
}

MeshTriPoint Mesh::toTriPoint( const PointOnFace & p ) const
{
    return toTriPoint( p.face, p.point );
}

MeshEdgePoint Mesh::toEdgePoint( EdgeId e, const Vector3f & p ) const
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

VertId Mesh::getClosestVertex( const PointOnFace & p ) const
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

UndirectedEdgeId Mesh::getClosestEdge( const PointOnFace & p ) const
{
    EdgeId e = topology.edgeWithLeft( p.face );
    Vector3f a, b, c;
    getLeftTriPoints( e, a, b, c );

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

Vector3f Mesh::triCenter( FaceId f ) const
{
    Vector3f v0, v1, v2;
    getTriPoints( f, v0, v1, v2 );
    return ( 1 / 3.0f ) * ( v0 + v1 + v2 );
}

Vector3f Mesh::leftDirDblArea( EdgeId e ) const
{
    VertId a, b, c;
    topology.getLeftTriVerts( e, a, b, c );
    assert( a.valid() && b.valid() && c.valid() );
    const auto & ap = points[a];
    const auto & bp = points[b];
    const auto & cp = points[c];
    return cross( bp - ap, cp - ap );
}

double Mesh::area( const FaceBitSet & fs ) const
{
    double twiceRes = 0;
    for ( auto f : fs )
        twiceRes += dblArea( f );
    return 0.5 * twiceRes;
}

Vector3f Mesh::dirDblArea( VertId v ) const
{
    Vector3f sum;
    for ( EdgeId e : orgRing( topology, v ) )
    {
        if ( topology.left( e ).valid() )
        {
            sum += leftDirDblArea( e );
        }
    }
    return sum;
}

Vector3f Mesh::normal( const MeshTriPoint & p ) const
{
    VertId a, b, c;
    topology.getLeftTriVerts( p.e, a, b, c );
    auto n0 = normal( a );
    auto n1 = normal( b );
    auto n2 = normal( c );
    return p.bary.interpolate( n0, n1, n2 );
}

Vector3f Mesh::pseudonormal( VertId v, const FaceBitSet * region ) const
{
    Vector3f sum;
    for ( EdgeId e : orgRing( topology, v ) )
    {
        const auto l = topology.left( e );
        if ( l && ( !region || region->test( l ) ) )
        {
            auto d0 = edgeVector( e );
            auto d1 = edgeVector( topology.next( e ) );
            auto angle = std::acos( std::clamp( dot( d0.normalized(), d1.normalized() ), -1.0f, 1.0f ) );
            auto n = cross( d0, d1 );
            assert( n.lengthSq() > 0.0f );
            sum += angle * n.normalized();
        }
    }

    return sum.normalized();
}

Vector3f Mesh::pseudonormal( EdgeId e, const FaceBitSet * region ) const
{
    auto l = topology.left( e );
    auto r = topology.right( e );
    if ( !l || ( region && !region->test( l ) ) )
    {
        auto n = normal( r );
        assert( n.lengthSq() > 0.0f );
        return n;
    }
    if ( !r || ( region && !region->test( r ) ) )
    {
        auto n = normal( l );
        assert( n.lengthSq() > 0.0f );
        return n;
    }
    auto nl = normal( l );
    auto nr = normal( r );
    assert( nl.lengthSq() > 0.0f );
    assert( nr.lengthSq() > 0.0f );
    return ( nl + nr ).normalized();
}

Vector3f Mesh::pseudonormal( const MeshTriPoint & p, const FaceBitSet * region ) const
{
    if ( auto v = p.inVertex( topology ) )
        return pseudonormal( v, region );
    if ( auto e = p.onEdge( topology ) )
        return pseudonormal( e->e, region );
    assert( !region || region->test( topology.left( p.e ) ) );
    auto n = leftNormal( p.e );
    assert( n.lengthSq() > 0.0f );
    return n;
}

float Mesh::signedDistance( const Vector3f & pt, const MeshTriPoint & proj, const FaceBitSet * region ) const
{
    const auto projPt = triPoint( proj );
    const float d = ( pt - projPt ).length();
    if ( dot( projPt - pt, pseudonormal( proj, region ) ) <= 0 )
        return d;
    else
        return -d;
}

float Mesh::signedDistance( const Vector3f & pt, const PointOnFace & proj, const FaceBitSet * region ) const
{
    return signedDistance( pt, toTriPoint( proj ), region ); 
}

float Mesh::signedDistance( const Vector3f & pt ) const
{
    auto res = signedDistance( pt, FLT_MAX );
    assert( res.has_value() );
    return *res;
}

std::optional<float> Mesh::signedDistance( const Vector3f & pt, float maxDistSq, const FaceBitSet * region ) const
{
    auto signRes = findSignedDistance( pt, { *this, region }, maxDistSq );
    if ( !signRes )
        return {};
    return signRes->dist;
}

float Mesh::sumAngles( VertId v, bool * outBoundaryVert ) const
{
    if ( outBoundaryVert )
        *outBoundaryVert = false;
    float sum = 0;
    for ( EdgeId e : orgRing( topology, v ) )
    {
        if ( topology.left( e ).valid() )
        {
            auto d0 = edgeVector( e );
            auto d1 = edgeVector( topology.next( e ) );
            auto angle = acos( std::clamp( dot( d0.normalized(), d1.normalized() ), -1.0f, 1.0f ) );
            sum += angle;
        }
        else if ( outBoundaryVert )
            *outBoundaryVert = true;
    }
    return sum;
}

VertBitSet Mesh::findSpikeVertices( float minSumAngle, const VertBitSet * region ) const
{
    const VertBitSet & testVerts = topology.getVertIds( region );
    VertBitSet res( testVerts.size() );
    BitSetParallelFor( testVerts, [&]( VertId v )
    {
        bool boundaryVert = false;
        auto a = sumAngles( v, &boundaryVert );
        if ( !boundaryVert && a < minSumAngle )
            res.set( v );
    } );
    return res;
}

float Mesh::dihedralAngleSin( EdgeId e ) const
{
    if ( topology.isBdEdge( e ) )
        return 0;
    auto leftNorm = leftNormal( e );
    auto rightNorm = leftNormal( e.sym() );
    auto edgeDir = edgeVector( e ).normalized();
    return dot( edgeDir, cross( leftNorm, rightNorm ) );
}

float Mesh::dihedralAngleCos( EdgeId e ) const
{
    if ( topology.isBdEdge( e ) )
        return 1;
    auto leftNorm = leftNormal( e );
    auto rightNorm = leftNormal( e.sym() );
    return dot( leftNorm, rightNorm );
}

class CreaseEdgesCalc 
{
public:
    CreaseEdgesCalc( const Mesh & mesh, float critCos ) : mesh_( mesh ), critCos_( critCos ) 
        { edges_.resize( mesh_.topology.undirectedEdgeSize() ); }
    CreaseEdgesCalc( CreaseEdgesCalc & x, tbb::split ) : mesh_( x.mesh_ ), critCos_( x.critCos_ )
        { edges_.resize( mesh_.topology.undirectedEdgeSize() ); }

    void join( const CreaseEdgesCalc & y ) { edges_ |= y.edges_; }

    UndirectedEdgeBitSet takeEdges() { return std::move( edges_ ); }

    void operator()( const tbb::blocked_range<UndirectedEdgeId> & r ) 
    {
        for ( UndirectedEdgeId ue = r.begin(); ue < r.end(); ++ue ) 
        {
            if ( mesh_.topology.isLoneEdge( ue ) )
                continue;
            auto dihedralCos = mesh_.dihedralAngleCos( ue );
            if ( dihedralCos <= critCos_ )
                edges_.set( ue );
        }
    }

private:
    const Mesh & mesh_;
    float critCos_ = 1;
    UndirectedEdgeBitSet edges_;
};

UndirectedEdgeBitSet Mesh::findCreaseEdges( float angleFromPlanar ) const
{
    assert( angleFromPlanar > 0 && angleFromPlanar < PI );
    const float critCos = std::cos( angleFromPlanar );
    CreaseEdgesCalc calc( *this, critCos );
    parallel_reduce( tbb::blocked_range<UndirectedEdgeId>( 0_ue, UndirectedEdgeId{ topology.undirectedEdgeSize() } ), calc );
    return calc.takeEdges();
}

float Mesh::leftCotan( EdgeId e ) const
{
    if ( !topology.left( e ).valid() )
        return 0;
    Vector3f p0, p1, p2;
    getLeftTriPoints( e, p0, p1, p2 );
    auto a = p0 - p2;
    auto b = p1 - p2;
    auto nom = dot( a, b );
    auto den = cross( a, b ).length();
    static constexpr float maxval = 1e5f;
    if ( fabs( nom ) >= maxval * den )
        return maxval * sgn( nom );
    return nom / den;
}

class VertBoundingBoxCalc 
{
public:
    VertBoundingBoxCalc( const Mesh & mesh, const AffineXf3f * toWorld ) : mesh_( mesh ), toWorld_( toWorld ) { }
    VertBoundingBoxCalc( VertBoundingBoxCalc & x, tbb::split ) : mesh_( x.mesh_ ), toWorld_( x.toWorld_ ) { }
    void join( const VertBoundingBoxCalc & y ) { box_.include( y.box_ ); }

    const Box3f & box() const { return box_; }

    void operator()( const tbb::blocked_range<VertId> & r ) 
    {
        for ( VertId v = r.begin(); v < r.end(); ++v ) 
        {
            if ( mesh_.topology.hasVert( v ) )
                box_.include( toWorld_ ? (*toWorld_)( mesh_.points[v] ) : mesh_.points[v] );
        }
    }
            
private:
    const Mesh & mesh_;
    const AffineXf3f * toWorld_ = nullptr;
    Box3f box_;
};

Box3f Mesh::computeBoundingBox( const AffineXf3f * toWorld ) const
{
    MR_TIMER
    const auto lastValidVert = topology.lastValidVert();

    VertBoundingBoxCalc calc( *this, toWorld );
    parallel_reduce( tbb::blocked_range<VertId>( 0_v, lastValidVert + 1 ), calc );
    return calc.box();
}

Box3f Mesh::getBoundingBox() const 
{ 
    return getAABBTree().getBoundingBox(); 
}

class FaceBoundingBoxCalc 
{
public:
    FaceBoundingBoxCalc( const Mesh& mesh, const FaceBitSet& region, const AffineXf3f* toWorld ) : mesh_( mesh ), region_( region ), toWorld_( toWorld ) {}
    FaceBoundingBoxCalc( FaceBoundingBoxCalc& x, tbb::split ) : mesh_( x.mesh_ ), region_( x.region_ ), toWorld_( x.toWorld_ ) {}
    void join( const FaceBoundingBoxCalc & y ) { box_.include( y.box_ ); }

    const Box3f & box() const { return box_; }

    void operator()( const tbb::blocked_range<FaceId> & r ) 
    {
        for ( FaceId f = r.begin(); f < r.end(); ++f ) 
        {
            if ( region_.test( f ) && mesh_.topology.hasFace( f ) )
            {
                for ( EdgeId e : leftRing( mesh_.topology, f ) )
                {
                    box_.include( toWorld_ ? ( *toWorld_ )( mesh_.points[mesh_.topology.org( e )] ) : mesh_.points[mesh_.topology.org( e )] );
                }
            }
        }
    }
            
private:
    const Mesh & mesh_;
    const FaceBitSet & region_;
    Box3f box_;
    const AffineXf3f* toWorld_ = nullptr;
};

Box3f Mesh::computeBoundingBox( const FaceBitSet * region, const AffineXf3f* toWorld ) const
{
    if ( !region )
        return computeBoundingBox( toWorld );

    MR_TIMER
    const auto lastValidFace = topology.lastValidFace();

    FaceBoundingBoxCalc calc( *this, *region, toWorld );
    parallel_reduce( tbb::blocked_range<FaceId>( 0_f, lastValidFace + 1 ), calc );
    return calc.box();
}

void Mesh::transform( const AffineXf3f & xf )
{
    MR_TIMER
    VertId lastValidVert = topology.lastValidVert();

    tbb::parallel_for(tbb::blocked_range<VertId>(VertId{ 0 }, lastValidVert + 1), [&](const tbb::blocked_range<VertId> & range)
    {
        for (VertId v = range.begin(); v < range.end(); ++v)
        {
            if (topology.hasVert(v))
                points[v] = xf(points[v]);
        }
    });
    invalidateCaches();
}

VertId Mesh::addPoint( const Vector3f & pos )
{
    VertId v = topology.addVertId();
    points.autoResizeAt( v ) = pos;
    return v;
}

EdgeId Mesh::addSeparateEdgeLoop( const std::vector<Vector3f>& contourPoints )
{
    if ( contourPoints.size() < 3 )
        return {};

    std::vector<EdgeId> newEdges = sMakeDisclosedEdgeLoop( *this, contourPoints);
    // close loop
    topology.splice( newEdges.front(), newEdges.back().sym() );

    return newEdges.front();
}

void Mesh::attachEdgeLoopPart( EdgeId first, EdgeId last, const std::vector<Vector3f>& contourPoints )
{
    if ( topology.left( first ) || topology.left( last ) )
    {
        assert( false );
        return;
    }
    if ( contourPoints.empty() )
        return;

    std::vector<EdgeId> newEdges = sMakeDisclosedEdgeLoop( *this, contourPoints );

    // connect with mesh
    auto firstConnectorEdge = topology.makeEdge();
    topology.splice( topology.prev( first.sym() ), firstConnectorEdge );
    topology.splice( newEdges.front(), firstConnectorEdge.sym() );

    topology.splice( last, newEdges.back().sym() );
}

VertId Mesh::splitEdge( EdgeId e, FaceBitSet * region )
{
    auto newPos = 0.5f * ( points[ topology.org( e ) ] + points[ topology.dest( e ) ] );
    VertId newv = topology.splitEdge( e, region );
    points.autoResizeAt( newv ) = newPos;
    return newv;
}

void Mesh::addPart( const Mesh & from,
    FaceMap * outFmap, VertMap * outVmap, EdgeMap * outEmap, bool rearrangeTriangles )
{
    MR_TIMER

    VertMap vmap;
    topology.addPart( from.topology, outFmap, &vmap, outEmap, rearrangeTriangles );
    if ( !vmap.empty() && vmap.back() >= points.size() )
        points.resize( vmap.back() + 1 );

    for ( VertId fromv{0}; fromv < vmap.size(); ++fromv )
    {
        VertId v = vmap[fromv];
        if ( v.valid() )
            points[v] = from.points[fromv];
    }

    if ( outVmap )
        *outVmap = std::move( vmap );
    invalidateCaches();
}

void Mesh::addPartByMask( const Mesh & from, const FaceBitSet & fromFaces,
    FaceMap * outFmap, VertMap * outVmap, EdgeMap * outEmap )
{
    HashToVectorMappingConverter m( from.topology, outFmap, outVmap, outEmap );
    addPartByMask( from, fromFaces, false, {}, {}, m.getPartMapping() );
}

void Mesh::addPartByMask( const Mesh & from, const FaceBitSet & fromFaces, const PartMapping & map )
{
    addPartByMask( from, fromFaces, false, {}, {}, map );
}

void Mesh::addPartByMask( const Mesh & from, const FaceBitSet & fromFaces, bool flipOrientation,
    const std::vector<std::vector<EdgeId>> & thisContours,
    const std::vector<std::vector<EdgeId>> & fromContours,
    FaceMap * outFmap, VertMap * outVmap, EdgeMap * outEmap )
{
    MR_TIMER
    HashToVectorMappingConverter m( from.topology, outFmap, outVmap, outEmap );
    addPartByMask( from, fromFaces, flipOrientation, thisContours, fromContours, m.getPartMapping() );
}

void Mesh::addPartByMask( const Mesh & from, const FaceBitSet & fromFaces, bool flipOrientation,
    const std::vector<std::vector<EdgeId>> & thisContours,
    const std::vector<std::vector<EdgeId>> & fromContours,
    PartMapping map )
{
    MR_TIMER

    VertHashMap localVmap;
    if ( !map.src2tgtVerts )
        map.src2tgtVerts = &localVmap;
    topology.addPartByMask( from.topology, fromFaces, flipOrientation, thisContours, fromContours, map );
    VertId lastPointId = topology.lastValidVert();
    if ( points.size() < lastPointId + 1 )
        points.resize( lastPointId + 1 );

    for ( const auto [ fromVert, thisVert ] : *map.src2tgtVerts )
        points[thisVert] = from.points[fromVert];

    invalidateCaches();
}

void Mesh::pack( FaceMap * outFmap, VertMap * outVmap, EdgeMap * outEmap, bool rearrangeTriangles )
{
    MR_TIMER

    if ( rearrangeTriangles )
        topology.rotateTriangles();
    Mesh packed;
    packed.addPart( *this, outFmap, outVmap, outEmap, rearrangeTriangles );
    *this = std::move( packed );
}

bool Mesh::intersectRay( const Vector3d& org, const Vector3d& dir, PointOnFace& res,
    double rayStart /*= 0.0f*/, double rayEnd /*= FLT_MAX */, const FaceBitSet* region /*= nullptr*/ ) const
{
    if( auto mir = rayMeshIntersect( { *this, region }, { org, dir }, rayStart, rayEnd ) )
    {
        res = mir->proj;
        return true;
    }
    else
    {
        return false;
    }
}
bool Mesh::intersectRay( const Vector3f& org, const Vector3f& dir, PointOnFace& res,
    float rayStart /*= 0.0f*/, float rayEnd /*= FLT_MAX */, const FaceBitSet* region /*= nullptr*/ ) const
{
    if( auto mir = rayMeshIntersect( { *this, region }, { org, dir }, rayStart, rayEnd ) )
    {
        res = mir->proj;
        return true;
    }
    else
    {
        return false;
    }
}

bool Mesh::intersectRay( const Vector3f& org, const Vector3f& dir, PointOnFace& res, const AffineXf3f& rayToMeshXf,
    float rayStart /*= 0*/, float rayEnd /*= FLT_MAX */, const FaceBitSet* region /*= nullptr*/ ) const
{
    if( auto mir = rayMeshIntersect( { *this, region }, { rayToMeshXf(org),rayToMeshXf.A * dir }, rayStart, rayEnd ) )
    {
        res = mir->proj;
        return true;
    }
    else
    {
        return false;
    }
}

bool Mesh::projectPoint( const Vector3f& point, PointOnFace& res, float maxDistSq, const FaceBitSet * region, const AffineXf3f * xf ) const
{
    auto proj = findProjection( point, { *this, region }, maxDistSq, xf );
    if ( !( proj.distSq < maxDistSq ) )
        return false;

    res = proj.proj;
    return true;
}

bool Mesh::projectPoint( const Vector3f& point, MeshProjectionResult& res, float maxDistSq, const FaceBitSet* region, const AffineXf3f * xf ) const
{
    auto proj = findProjection( point, { *this, region }, maxDistSq, xf );
    if (!(proj.distSq < maxDistSq))
        return false;

    res = proj;
    return true;
}

std::optional<MeshProjectionResult> Mesh::projectPoint( const Vector3f& point, float maxDistSq, const FaceBitSet * region, const AffineXf3f * xf ) const
{
    auto proj = findProjection( point, { *this, region }, maxDistSq, xf );
    if ( !( proj.distSq < maxDistSq ) )
        return {};

    return proj;
}

const AABBTree & Mesh::getAABBTree() const 
{ 
    const auto & res = AABBTreeOwner_.getOrCreate( [this]{ return AABBTree( *this ); } );
    assert( res.containsSameNumberOfTris( *this ) );
    return res;
}

void Mesh::invalidateCaches()
{
    AABBTreeOwner_.reset();
}

Vector3f Mesh::findCenterFromPoints() const
{
    Vector3f res;
    int count = 0;
    const auto & validPoints = topology.edgePerVertex();
    for (auto edge : validPoints)
    {
        if(edge.valid())
        {
            res += points[topology.org(edge)];
            count++;
        }
    }
    assert(count > 0);
    return res / float(count);
}

Vector3f Mesh::findCenterFromFaces() const
{
    Vector3f acc(0., 0., 0.);
    auto &edgePerFaces = topology.edgePerFace();
    float triAreaAcc = 0;
    for (auto edge : edgePerFaces)
    {
        if (edge.valid())
        {
            VertId v0, v1, v2;
            topology.getLeftTriVerts(edge, v0, v1, v2);
            //area of triangle corresponds to the weight of each point
            float triArea = leftDirDblArea(edge).length();
            Vector3f center = points[v0] + points[v1] + points[v2];
            acc += center * triArea;
            triAreaAcc += triArea;
        }
    }
    assert(triAreaAcc > 0.f);
    return acc / triAreaAcc / 3.f;
}

Vector3f Mesh::findCenterFromBBox() const
{
    return computeBoundingBox().center();
}

} //namespace MR
