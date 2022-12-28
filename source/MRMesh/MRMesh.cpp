#include "MRMesh.h"
#include "MRAABBTree.h"
#include "MRAffineXf3.h"
#include "MRBitSet.h"
#include "MRBitSetParallelFor.h"
#include "MRBox.h"
#include "MRComputeBoundingBox.h"
#include "MRConstants.h"
#include "MRCube.h"
#include "MREdgeIterator.h"
#include "MRGTest.h"
#include "MRLine3.h"
#include "MRLineSegm.h"
#include "MRMeshBuilder.h"
#include "MRMeshIntersect.h"
#include "MRMeshTriPoint.h"
#include "MROrder.h"
#include "MRQuadraticForm.h"
#include "MRRegionBoundary.h"
#include "MRRingIterator.h"
#include "MRTimer.h"
#include "MRTriangleIntersection.h"
#include "MRTriMath.h"
#include "MRPch/MRTBB.h"

namespace MR
{

namespace
{

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

Mesh Mesh::fromTriangles(
    VertCoords vertexCoordinates,
    const Triangulation & t, const MeshBuilder::BuildSettings & settings )
{
    MR_TIMER
    Mesh res;
    res.points = std::move( vertexCoordinates );
    res.topology = MeshBuilder::fromTriangles( t, settings );
    return res;
}

Mesh Mesh::fromTrianglesDuplicatingNonManifoldVertices( 
    VertCoords vertexCoordinates,
    Triangulation & t,
    std::vector<MeshBuilder::VertDuplication> * dups,
    const MeshBuilder::BuildSettings & settings )
{
    MR_TIMER
    Mesh res;
    res.points = std::move( vertexCoordinates );
    std::vector<MeshBuilder::VertDuplication> localDups;
    res.topology = MeshBuilder::fromTrianglesDuplicatingNonManifoldVertices( t, &localDups, settings );
    res.points.resize( res.topology.vertSize() );
    for ( const auto & d : localDups )
        res.points[d.dupVert] = res.points[d.srcVert];
    if ( dups )
        *dups = std::move( localDups );
    return res;
}

bool Mesh::operator ==( const Mesh & b ) const
{
    MR_TIMER
    if ( topology != b.topology )
        return false;
    for ( auto v : topology.getValidVerts() )
        if ( points[v] != b.points[v] )
            return false;
    return true;
}

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
    MR_TIMER
    constexpr int NUM_PARTS = 64; // independent on hardware concurrency
    const auto facesInPart = topology.faceSize() / NUM_PARTS;
    if ( facesInPart < 16 )
    {
        double twiceRes = 0;
        for ( auto f : fs )
            twiceRes += dblArea( f );
        return 0.5 * twiceRes;
    }

    double partDblArea[NUM_PARTS] = {};
    tbb::parallel_for( tbb::blocked_range( 0, NUM_PARTS ),
        [&]( const tbb::blocked_range<int> & range )
    {
        for ( int part = range.begin(); part < range.end(); ++part )
        {
            double myDblArea = 0;
            const FaceId fBeg( part * facesInPart );
            const FaceId fEnd( part + 1 < NUM_PARTS ? ( part + 1 ) * facesInPart : topology.faceSize() );
            for ( auto f = fBeg; f < fEnd; ++f )
                if ( fs.test( f ) )
                    myDblArea += dblArea( f );
            partDblArea[part] = myDblArea;
        }
    } );

    double twiceRes = 0;
    for ( auto da : partDblArea )
        twiceRes += da;
    return 0.5 * twiceRes;
}

class FaceVolumeCalc
{
public:
    FaceVolumeCalc( const Mesh& mesh, const FaceBitSet& region) : mesh_( mesh ), region_( region )
    {}
    FaceVolumeCalc( FaceVolumeCalc& x, tbb::split ) : mesh_( x.mesh_ ), region_( x.region_ )
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
            if ( region_.test( f ) && mesh_.topology.hasFace( f ) )
            {
                Vector3f coords[3];
                mesh_.getTriPoints( f, coords );
                volume_ += mixed( coords[0], coords[1], coords[2] );
            }
        }
    }

private:
    const Mesh& mesh_;
    const FaceBitSet& region_;
    double volume_{ 0.0 };
};

double Mesh::volume( const FaceBitSet* region /*= nullptr */ ) const
{
    if ( !topology.isClosed( region ) )
        return DBL_MAX;

    MR_TIMER
    const auto lastValidFace = topology.lastValidFace();
    const auto& faces = topology.getFaceIds( region );
    FaceVolumeCalc calc( *this, faces );
    parallel_reduce( tbb::blocked_range<FaceId>( 0_f, lastValidFace + 1 ), calc );
    return calc.volume() / 6.0;
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
            auto angle = MR::angle( d0, d1 );
            auto n = cross( d0, d1 );
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
            auto angle = MR::angle( d0, d1 );
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
    return MR::dihedralAngleSin( leftNormal( e ), leftNormal( e.sym() ), edgeVector( e ) );
}

float Mesh::dihedralAngleCos( EdgeId e ) const
{
    if ( topology.isBdEdge( e ) )
        return 1;
    return MR::dihedralAngleCos( leftNormal( e ), leftNormal( e.sym() ) );
}

float Mesh::dihedralAngle( EdgeId e ) const
{
    if ( topology.isBdEdge( e ) )
        return 0;
    return MR::dihedralAngle( leftNormal( e ), leftNormal( e.sym() ), edgeVector( e ) );
}

float Mesh::discreteMeanCurvature( VertId v ) const
{
    float sumArea = 0;
    float sumAngLen = 0;
    for ( EdgeId e : orgRing( topology, v ) )
    {
        auto l = topology.left( e );
        if ( !l )
            continue; // area( l ) is not defined and dihedralAngle( e ) = 0
        sumArea += area( l );
        sumAngLen += dihedralAngle( e ) * edgeLength( e );
    }
    // sumAngLen / (2*2) because of mean curvature definition * each edge has 2 vertices,
    // sumArea / 3 because each triangle has 3 vertices
    return ( sumArea > 0 ) ? 0.75f * sumAngLen / sumArea : 0;
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

QuadraticForm3f Mesh::quadraticForm( VertId v, const FaceBitSet * region ) const
{
    QuadraticForm3f qf;
    for ( EdgeId e : orgRing( topology, v ) )
    {
        if ( topology.isBdEdge( e, region ) )
            qf.addDistToLine( edgeVector( e ).normalized() );
        if ( topology.isLeftInRegion( e, region ) )
            qf.addDistToPlane( leftNormal( e ) );
    }
    return qf;
}

Box3f Mesh::computeBoundingBox( const AffineXf3f * toWorld ) const
{
    return MR::computeBoundingBox( points, topology.getValidVerts(), toWorld );
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

float Mesh::averageEdgeLength() const
{
    MR_TIMER
    double sum = 0;
    int n = 0;
    for ( auto ue : undirectedEdges( topology ) )
    {
        sum += edgeLength( ue );
        ++n;
    }
    return n > 0 ? float( sum / n ) : 0.0f;
}

void Mesh::zeroUnusedPoints()
{
    MR_TIMER

    tbb::parallel_for( tbb::blocked_range<VertId>( 0_v, VertId{ points.size() } ), [&] ( const tbb::blocked_range<VertId>& range )
    {
        for ( VertId v = range.begin(); v < range.end(); ++v )
        {
            if ( !topology.hasVert( v ) )
                points[v] = {};
        }
    } );
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

    invalidateCaches();

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

    invalidateCaches();
}

EdgeId Mesh::splitEdge( EdgeId e, const Vector3f & newVertPos, FaceBitSet * region )
{
    EdgeId newe = topology.splitEdge( e, region );
    points.autoResizeAt( topology.org( e ) ) = newVertPos;
    return newe;
}

VertId Mesh::splitFace( FaceId f, FaceBitSet * region )
{
    auto newPos = triCenter( f );
    VertId newv = topology.splitFace( f, region );
    points.autoResizeAt( newv ) = newPos;
    return newv;
}

void Mesh::addPart( const Mesh & from,
    FaceMap * outFmap, VertMap * outVmap, WholeEdgeMap * outEmap, bool rearrangeTriangles )
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

void Mesh::addPartByMask( const Mesh & from, const FaceBitSet & fromFaces, const PartMapping & map )
{
    addPartByMask( from, fromFaces, false, {}, {}, map );
}

void Mesh::addPartByMask( const Mesh & from, const FaceBitSet & fromFaces, bool flipOrientation,
    const std::vector<std::vector<EdgeId>> & thisContours,
    const std::vector<std::vector<EdgeId>> & fromContours,
    const PartMapping & map )
{
    MR_TIMER
    addPartBy( from, begin( fromFaces ), end( fromFaces ), fromFaces.count(), flipOrientation, thisContours, fromContours, map );
}

void Mesh::addPartByFaceMap( const Mesh & from, const FaceMap & fromFaces, bool flipOrientation,
    const std::vector<std::vector<EdgeId>> & thisContours,
    const std::vector<std::vector<EdgeId>> & fromContours,
    const PartMapping & map )
{
    MR_TIMER
    addPartBy( from, begin( fromFaces ), end( fromFaces ), fromFaces.size(), flipOrientation, thisContours, fromContours, map );
}

template<typename I>
void Mesh::addPartBy( const Mesh & from, I fbegin, I fend, size_t fcount, bool flipOrientation,
    const std::vector<std::vector<EdgeId>> & thisContours,
    const std::vector<std::vector<EdgeId>> & fromContours,
    PartMapping map )
{
    MR_TIMER

    VertHashMap localVmap;
    if ( !map.src2tgtVerts )
        map.src2tgtVerts = &localVmap;
    topology.addPartBy( from.topology, fbegin, fend, fcount, flipOrientation, thisContours, fromContours, map );
    VertId lastPointId = topology.lastValidVert();
    if ( points.size() < lastPointId + 1 )
        points.resize( lastPointId + 1 );

    for ( const auto & [ fromVert, thisVert ] : *map.src2tgtVerts )
        points[thisVert] = from.points[fromVert];

    invalidateCaches();
}

template MRMESH_API void Mesh::addPartBy( const Mesh & from,
    SetBitIteratorT<FaceBitSet> fbegin, SetBitIteratorT<FaceBitSet> fend, size_t fcount, bool flipOrientation,
    const std::vector<std::vector<EdgeId>> & thisContours,
    const std::vector<std::vector<EdgeId>> & fromContours,
    PartMapping map );
template MRMESH_API void Mesh::addPartBy( const Mesh & from,
    FaceMap::iterator fbegin, FaceMap::iterator fend, size_t fcount, bool flipOrientation,
    const std::vector<std::vector<EdgeId>> & thisContours,
    const std::vector<std::vector<EdgeId>> & fromContours,
    PartMapping map );

Mesh Mesh::cloneRegion( const FaceBitSet & region, bool flipOrientation, const PartMapping & map ) const
{
    MR_TIMER

    Mesh res;
    const auto fcount = region.count();
    res.topology.faceReserve( fcount );
    const auto vcount = getIncidentVerts( topology, region ).count();
    res.topology.vertReserve( vcount );
    const auto ecount = 2 * getIncidentEdges( topology, region ).count();
    res.topology.edgeReserve( ecount );

    res.addPartByMask( *this, region, flipOrientation, {}, {}, map );

    assert( res.topology.faceSize() == fcount );
    assert( res.topology.faceCapacity() == fcount );
    assert( res.topology.vertSize() == vcount );
    assert( res.topology.vertCapacity() == vcount );
    assert( res.topology.edgeSize() == ecount );
    assert( res.topology.edgeCapacity() == ecount );
    return res;
}

void Mesh::pack( FaceMap * outFmap, VertMap * outVmap, WholeEdgeMap * outEmap, bool rearrangeTriangles )
{
    MR_TIMER

    if ( rearrangeTriangles )
        topology.rotateTriangles();
    Mesh packed;
    packed.addPart( *this, outFmap, outVmap, outEmap, rearrangeTriangles );
    *this = std::move( packed );
}

PackMapping Mesh::packOptimally( bool preserveAABBTree )
{
    MR_TIMER

    PackMapping map;
    if ( preserveAABBTree )
    {
        getAABBTree(); // ensure that tree is constructed
        map.f.b.resize( topology.faceSize() );
        const bool packed = topology.numValidFaces() == topology.faceSize();
        if ( !packed )
        {
            for ( FaceId f = 0_f; f < map.f.b.size(); ++f )
                if ( !topology.hasFace( f ) )
                    map.f.b[f] = FaceId{};
        }
        AABBTreeOwner_.get()->getLeafOrderAndReset( map.f );
    }
    else
    {
        AABBTreeOwner_.reset();
        map.f = getOptimalFaceOrdering( *this );
    }
    map.v = getVertexOrdering( map.f, topology );
    map.e = getEdgeOrdering( map.f, topology );
    topology.pack( map );

    VertCoords newPoints( map.v.tsize );
    tbb::parallel_for( tbb::blocked_range( 0_v, VertId{ map.v.b.size() } ),
        [&]( const tbb::blocked_range<VertId> & range )
    {
        for ( auto oldv = range.begin(); oldv < range.end(); ++oldv )
        {
            auto newv = map.v.b[oldv];
            if ( !newv )
                continue;
            newPoints[newv] = points[oldv];
        }
    } );
    points = std::move( newPoints );
    return map;
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

size_t Mesh::heapBytes() const
{
    return topology.heapBytes()
        + points.heapBytes()
        + AABBTreeOwner_.heapBytes();
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

void Mesh::mirror( const Plane3f& plane )
{
    for ( auto& p : points )
    {
        p += 2.0f * ( plane.project( p ) - p );
    }

    topology.flipOrientation();
    invalidateCaches();
}

TEST( MRMesh, BasicExport )
{
    Mesh mesh = makeCube();

    const std::vector<ThreeVertIds> triangles = mesh.topology.getAllTriVerts();

    const std::vector<Vector3f> & points =  mesh.points.vec_;
    const int * vertexTripples = reinterpret_cast<const int*>( triangles.data() );

    (void)points;
    (void)vertexTripples;
}

TEST(MRMesh, SplitEdge) 
{
    Triangulation t{
        { VertId{0}, VertId{1}, VertId{2} },
        { VertId{0}, VertId{2}, VertId{3} }
    };
    Mesh mesh;
    mesh.topology = MeshBuilder::fromTriangles( t );
    mesh.points.emplace_back( 0.f, 0.f, 0.f );
    mesh.points.emplace_back( 1.f, 0.f, 0.f );
    mesh.points.emplace_back( 1.f, 1.f, 0.f );
    mesh.points.emplace_back( 0.f, 1.f, 0.f );

    EXPECT_EQ( mesh.topology.numValidVerts(), 4 );
    EXPECT_EQ( mesh.points.size(), 4 );
    EXPECT_EQ( mesh.topology.numValidFaces(), 2 );
    EXPECT_EQ( mesh.topology.lastNotLoneEdge(), EdgeId(9) ); // 5*2 = 10 half-edges in total

    FaceBitSet region( 2 );
    region.set( 0_f );

    auto e02 = mesh.topology.findEdge( VertId{0}, VertId{2} );
    EXPECT_TRUE( e02.valid() );
    auto ex = mesh.splitEdge( e02, &region );
    VertId v02 = mesh.topology.org( e02 );
    EXPECT_EQ( mesh.topology.dest( ex ), v02 );
    EXPECT_EQ( mesh.topology.numValidVerts(), 5 );
    EXPECT_EQ( mesh.points.size(), 5 );
    EXPECT_EQ( mesh.topology.numValidFaces(), 4 );
    EXPECT_EQ( mesh.topology.lastNotLoneEdge(), EdgeId(15) ); // 8*2 = 16 half-edges in total
    EXPECT_EQ( mesh.points[v02], ( Vector3f(.5f, .5f, 0.f) ) );
    EXPECT_EQ( region.count(), 2 );

    auto e01 = mesh.topology.findEdge( VertId{0}, VertId{1} );
    EXPECT_TRUE( e01.valid() );
    auto ey = mesh.splitEdge( e01, &region );
    VertId v01 =  mesh.topology.org( e01 );
    EXPECT_EQ( mesh.topology.dest( ey ), v01 );
    EXPECT_EQ( mesh.topology.numValidVerts(), 6 );
    EXPECT_EQ( mesh.points.size(), 6 );
    EXPECT_EQ( mesh.topology.numValidFaces(), 5 );
    EXPECT_EQ( mesh.topology.lastNotLoneEdge(), EdgeId(19) ); // 10*2 = 20 half-edges in total
    EXPECT_EQ( mesh.points[v01], ( Vector3f(.5f, 0.f, 0.f) ) );
    EXPECT_EQ( region.count(), 3 );
}

TEST(MRMesh, SplitEdge1) 
{
    Mesh mesh;
    const auto e01 = mesh.topology.makeEdge();
    mesh.topology.setOrg( e01, mesh.topology.addVertId() );
    mesh.topology.setOrg( e01.sym(), mesh.topology.addVertId() );
    mesh.points.emplace_back( 0.f, 0.f, 0.f );
    mesh.points.emplace_back( 1.f, 0.f, 0.f );

    EXPECT_EQ( mesh.topology.numValidVerts(), 2 );
    EXPECT_EQ( mesh.points.size(), 2 );
    EXPECT_EQ( mesh.topology.lastNotLoneEdge(), EdgeId(1) ); // 1*2 = 2 half-edges in total

    auto ey = mesh.splitEdge( e01 );
    VertId v01 =  mesh.topology.org( e01 );
    EXPECT_EQ( mesh.topology.dest( ey ), v01 );
    EXPECT_EQ( mesh.topology.numValidVerts(), 3 );
    EXPECT_EQ( mesh.points.size(), 3 );
    EXPECT_EQ( mesh.topology.lastNotLoneEdge(), EdgeId(3) ); // 2*2 = 4 half-edges in total
    EXPECT_EQ( mesh.points[v01], ( Vector3f( .5f, 0.f, 0.f ) ) );
}

TEST(MRMesh, SplitFace) 
{
    Triangulation t{
        { VertId{0}, VertId{1}, VertId{2} }
    };
    Mesh mesh;
    mesh.topology = MeshBuilder::fromTriangles( t );
    mesh.points.emplace_back( 0.f, 0.f, 0.f );
    mesh.points.emplace_back( 0.f, 0.f, 1.f );
    mesh.points.emplace_back( 0.f, 1.f, 0.f );

    EXPECT_EQ( mesh.topology.numValidVerts(), 3 );
    EXPECT_EQ( mesh.points.size(), 3 );
    EXPECT_EQ( mesh.topology.numValidFaces(), 1 );
    EXPECT_EQ( mesh.topology.lastNotLoneEdge(), EdgeId(5) ); // 3*2 = 6 half-edges in total

    mesh.splitFace( 0_f );
    EXPECT_EQ( mesh.topology.numValidVerts(), 4 );
    EXPECT_EQ( mesh.points.size(), 4 );
    EXPECT_EQ( mesh.topology.numValidFaces(), 3 );
    EXPECT_EQ( mesh.topology.lastNotLoneEdge(), EdgeId(11) ); // 6*2 = 12 half-edges in total
}

} //namespace MR
