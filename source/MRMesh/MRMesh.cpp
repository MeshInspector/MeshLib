#include "MRMesh.h"
#include "MRAABBTree.h"
#include "MRAABBTreePoints.h"
#include "MRAffineXf3.h"
#include "MRBitSet.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
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
#include "MRIdentifyVertices.h"
#include "MRMeshFillHole.h"
#include "MRTriMesh.h"
#include "MRDipole.h"
#include "MRPch/MRTBB.h"

namespace MR
{

namespace
{

// makes edge path connecting all points, but not the first with the last
EdgePath sMakeEdgePath( Mesh& mesh, const std::vector<Vector3f>& contourPoints )
{
    EdgePath newEdges( contourPoints.size() );
    for ( int i = 0; i < contourPoints.size(); ++i )
    {
        auto newVert = mesh.addPoint( contourPoints[i] );
        newEdges[i] = mesh.topology.makeEdge();
        mesh.topology.setOrg( newEdges[i], newVert );
    }
    for ( int i = 0; i + 1 < newEdges.size(); ++i )
    {
        mesh.topology.splice( newEdges[i + 1], newEdges[i].sym() );
    }
    return newEdges;
}
}

Mesh Mesh::fromTriangles(
    VertCoords vertexCoordinates,
    const Triangulation& t, const MeshBuilder::BuildSettings& settings, ProgressCallback cb /*= {}*/ )
{
    MR_TIMER;
    Mesh res;
    res.points = std::move( vertexCoordinates );
    res.topology = MeshBuilder::fromTriangles( t, settings, cb );
    return res;
}

Mesh Mesh::fromTriMesh(
    TriMesh && triMesh,
    const MeshBuilder::BuildSettings& settings, ProgressCallback cb  )
{
    return fromTriangles( std::move( triMesh.points ), triMesh.tris, settings, cb );
}

Mesh Mesh::fromTrianglesDuplicatingNonManifoldVertices(
    VertCoords vertexCoordinates,
    Triangulation & t,
    std::vector<MeshBuilder::VertDuplication> * dups,
    const MeshBuilder::BuildSettings & settings )
{
    MR_TIMER;
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

Mesh Mesh::fromFaceSoup(
    VertCoords vertexCoordinates,
    const std::vector<VertId> & verts, const Vector<MeshBuilder::VertSpan, FaceId> & faces,
    const MeshBuilder::BuildSettings& settings, ProgressCallback cb /*= {}*/ )
{
    MR_TIMER;
    Mesh res;
    res.points = std::move( vertexCoordinates );
    res.topology = MeshBuilder::fromFaceSoup( verts, faces, settings, subprogress( cb, 0.0f, 0.8f ) );

    struct FaceFill
    {
        HoleFillPlan plan;
        EdgeId e; // fill left of it
    };
    std::vector<FaceFill> faceFills;
    for ( auto f : res.topology.getValidFaces() )
    {
        auto e = res.topology.edgeWithLeft( f );
        if ( !res.topology.isLeftTri( e ) )
            faceFills.push_back( { {}, e } );
    }

    ParallelFor( faceFills, [&]( size_t i )
    {
        faceFills[i].plan = getPlanarHoleFillPlan( res, faceFills[i].e );
    }, subprogress( cb, 0.8f, 0.9f ) );

    for ( auto & x : faceFills )
        executeHoleFillPlan( res, x.e, x.plan );

    reportProgress( cb, 1.0f );

    return res;
}

Mesh Mesh::fromPointTriples( const std::vector<Triangle3f> & posTriples, bool duplicateNonManifoldVertices )
{
    MR_TIMER;
    MeshBuilder::VertexIdentifier vi;
    vi.reserve( posTriples.size() );
    vi.addTriangles( posTriples );
    if ( duplicateNonManifoldVertices )
    {
        auto t = vi.takeTriangulation();
        return fromTrianglesDuplicatingNonManifoldVertices( vi.takePoints(), t );
    }
    return fromTriangles( vi.takePoints(), vi.takeTriangulation() );
}

bool Mesh::operator ==( const Mesh & b ) const
{
    MR_TIMER;
    if ( topology != b.topology )
        return false;
    for ( auto v : topology.getValidVerts() )
        if ( points[v] != b.points[v] )
            return false;
    return true;
}

MeshTriPoint Mesh::toTriPoint( VertId v ) const
{
    return MeshTriPoint( topology, v );
}

MeshEdgePoint Mesh::toEdgePoint( VertId v ) const
{
    return MeshEdgePoint( topology, v );
}

bool Mesh::isOutsideByProjNorm( const Vector3f & pt, const MeshProjectionResult & proj, const FaceBitSet * region ) const
{
    return dot( proj.proj.point - pt, pseudonormal( proj.mtp, region ) ) <= 0;
}

float Mesh::signedDistance( const Vector3f & pt, const MeshProjectionResult & proj, const FaceBitSet * region ) const
{
    if ( isOutsideByProjNorm( pt, proj, region ) )
        return std::sqrt( proj.distSq );
    else
        return -std::sqrt( proj.distSq );
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

float Mesh::calcFastWindingNumber( const Vector3f & pt, float beta ) const
{
    return MR::calcFastWindingNumber( getDipoles(), getAABBTree(), *this, pt, beta, {} );
}

QuadraticForm3f Mesh::quadraticForm( VertId v, bool angleWeigted, const FaceBitSet * region, const UndirectedEdgeBitSet * creases ) const
{
    return MR::quadraticForm( topology, points, v, angleWeigted, region, creases );
}

Box3f Mesh::computeBoundingBox( const AffineXf3f * toWorld ) const
{
    return MR::computeBoundingBox( points, topology.getValidVerts(), toWorld );
}

Box3f Mesh::getBoundingBox() const
{
    return getAABBTree().getBoundingBox();
}

Box3f Mesh::computeBoundingBox( const FaceBitSet * region, const AffineXf3f* toWorld ) const
{
    return MR::computeBoundingBox( topology, points, region, toWorld );
}

void Mesh::zeroUnusedPoints()
{
    MR_TIMER;

    tbb::parallel_for( tbb::blocked_range<VertId>( 0_v, VertId{ points.size() } ), [&] ( const tbb::blocked_range<VertId>& range )
    {
        for ( VertId v = range.begin(); v < range.end(); ++v )
        {
            if ( !topology.hasVert( v ) )
                points[v] = {};
        }
    } );
}

void Mesh::transform( const AffineXf3f& xf, const VertBitSet* region )
{
    MR_TIMER;

    BitSetParallelFor( topology.getVertIds( region ), [&] ( const VertId v )
    {
        points[v] = xf( points[v] );
    } );
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

    auto newEdges = sMakeEdgePath( *this, contourPoints);
    // close loop
    topology.splice( newEdges.front(), newEdges.back().sym() );

    invalidateCaches();

    return newEdges.front();
}

EdgeId Mesh::addSeparateContours( const Contours3f& contours, const AffineXf3f* xf )
{
    EdgeId firstNewEdge;
    for ( const auto& cont : contours )
    {
        bool closed = cont.size() > 2 && cont.front() == cont.back();
        size_t numNewVerts = closed ? cont.size() - 1 : cont.size();
        size_t numNewEdges = cont.size() - 1;
        EdgeId prevEdgeId;
        EdgeId firstContEdge;
        for ( size_t i = 0; i < numNewVerts; ++i )
        {
            auto newVert = addPoint( xf ? ( *xf )( cont[i] ) : cont[i] );
            if ( prevEdgeId )
                topology.setOrg( prevEdgeId.sym(), newVert );
            if ( i < numNewEdges )
            {
                auto newEdge = topology.makeEdge();
                if ( !firstContEdge )
                {
                    firstContEdge = newEdge;
                    if ( !firstNewEdge )
                        firstNewEdge = firstContEdge;
                }
                if ( prevEdgeId )
                    topology.splice( prevEdgeId.sym(), newEdge );
                else
                    topology.setOrg( newEdge, newVert );
                prevEdgeId = newEdge;
            }
        }
        if ( closed )
            topology.splice( firstContEdge, prevEdgeId.sym() );
    }

    invalidateCaches();

    return firstNewEdge;
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

    auto newEdges = sMakeEdgePath( *this, contourPoints );

    // connect with mesh
    auto firstConnectorEdge = topology.makeEdge();
    topology.splice( topology.prev( first.sym() ), firstConnectorEdge );
    topology.splice( newEdges.front(), firstConnectorEdge.sym() );

    topology.splice( last, newEdges.back().sym() );

    invalidateCaches();
}

EdgeId Mesh::splitEdge( EdgeId e, const Vector3f & newVertPos, FaceBitSet * region, FaceHashMap * new2Old )
{
    EdgeId newe = topology.splitEdge( e, region, new2Old );
    points.autoResizeAt( topology.org( e ) ) = newVertPos;
    return newe;
}

VertId Mesh::splitFace( FaceId f, const Vector3f & newVertPos, FaceBitSet * region, FaceHashMap * new2Old )
{
    VertId newv = topology.splitFace( f, region, new2Old );
    points.autoResizeAt( newv ) = newVertPos;
    return newv;
}

void Mesh::addMesh( const Mesh & from,
    FaceMap * outFmap, VertMap * outVmap, WholeEdgeMap * outEmap, bool rearrangeTriangles )
{
    addMesh( from, Src2TgtMaps( outFmap, outVmap, outEmap ), rearrangeTriangles );
}

void Mesh::addMesh( const Mesh & from, PartMapping map, bool rearrangeTriangles )
{
    MR_TIMER;

    invalidateCaches();

    auto localVmap = VertMapOrHashMap::createMap();
    if ( !map.src2tgtVerts )
        map.src2tgtVerts = &localVmap;
    topology.addPart( from.topology, map, rearrangeTriangles );
    VertId lastPointId = topology.lastValidVert();
    if ( points.size() < lastPointId + 1 )
        points.resize( lastPointId + 1 );
    map.src2tgtVerts->forEach( [&]( VertId fromVert, VertId thisVert ) { points[thisVert] = from.points[fromVert]; } );
}

void Mesh::addMeshPart( const MeshPart & from, const PartMapping & map )
{
    addMeshPart( from, false, {}, {}, map );
}

void Mesh::addMeshPart( const MeshPart & from, bool flipOrientation,
    const std::vector<EdgePath> & thisContours,
    const std::vector<EdgePath> & fromContours,
    const PartMapping & map )
{
    MR_TIMER;
    const auto & fromFaces = from.mesh.topology.getFaceIds( from.region );
    addPartBy( from.mesh, begin( fromFaces ), end( fromFaces ), fromFaces.count(), flipOrientation, thisContours, fromContours, map );
}

void Mesh::addPartByFaceMap( const Mesh & from, const FaceMap & fromFaces, bool flipOrientation,
    const std::vector<EdgePath> & thisContours,
    const std::vector<EdgePath> & fromContours,
    const PartMapping & map )
{
    MR_TIMER;
    addPartBy( from, begin( fromFaces ), end( fromFaces ), fromFaces.size(), flipOrientation, thisContours, fromContours, map );
}

template<typename I>
void Mesh::addPartBy( const Mesh & from, I fbegin, I fend, size_t fcount, bool flipOrientation,
    const std::vector<EdgePath> & thisContours,
    const std::vector<EdgePath> & fromContours,
    PartMapping map )
{
    MR_TIMER;

    invalidateCaches();

    auto localVmap = VertMapOrHashMap::createHashMap();
    if ( !map.src2tgtVerts )
        map.src2tgtVerts = &localVmap;
    topology.addPartBy( from.topology, fbegin, fend, fcount, flipOrientation, thisContours, fromContours, map );
    VertId lastPointId = topology.lastValidVert();
    if ( points.size() < lastPointId + 1 )
        points.resize( lastPointId + 1 );
    map.src2tgtVerts->forEach( [&]( VertId fromVert, VertId thisVert ) { points[thisVert] = from.points[fromVert]; } );
}

template MRMESH_API void Mesh::addPartBy( const Mesh & from,
    SetBitIteratorT<FaceBitSet> fbegin, SetBitIteratorT<FaceBitSet> fend, size_t fcount, bool flipOrientation,
    const std::vector<EdgePath> & thisContours,
    const std::vector<EdgePath> & fromContours,
    PartMapping map );
template MRMESH_API void Mesh::addPartBy( const Mesh & from,
    FaceMap::iterator fbegin, FaceMap::iterator fend, size_t fcount, bool flipOrientation,
    const std::vector<EdgePath> & thisContours,
    const std::vector<EdgePath> & fromContours,
    PartMapping map );

Mesh Mesh::cloneRegion( const FaceBitSet & region, bool flipOrientation, const PartMapping & map ) const
{
    MR_TIMER;

    Mesh res;
    const auto fcount = region.count();
    res.topology.faceReserve( fcount );
    const auto vcount = getIncidentVerts( topology, region ).count();
    res.topology.vertReserve( vcount );
    const auto ecount = 2 * getIncidentEdges( topology, region ).count();
    res.topology.edgeReserve( ecount );

    res.addMeshPart( { *this, &region }, flipOrientation, {}, {}, map );

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
    MR_TIMER;

    if ( rearrangeTriangles )
        topology.rotateTriangles();
    Mesh packed;
    packed.points.reserve( topology.numValidVerts() );
    packed.topology.vertReserve( topology.numValidVerts() );
    packed.topology.faceReserve( topology.numValidFaces() );
    packed.topology.edgeReserve( 2 * topology.computeNotLoneUndirectedEdges() );
    packed.addMesh( *this, outFmap, outVmap, outEmap, rearrangeTriangles );
    *this = std::move( packed );
}

Expected<void> Mesh::pack( const PackMapping & map, ProgressCallback cb )
{
    MR_TIMER;
    topology.pack( map );
    if ( !reportProgress( cb, 0.8f ) )
        return unexpectedOperationCanceled();

    VertCoords newPoints( map.v.tsize );
    if ( !ParallelFor( 0_v, VertId{ map.v.b.size() }, [&]( VertId oldv )
    {
        if ( auto newv = map.v.b[oldv] )
            newPoints[newv] = points[oldv];
    }, subprogress( cb, 0.8f, 1.0f ) ) )
        return unexpectedOperationCanceled();
    points = std::move( newPoints );
    return {};
}

PackMapping Mesh::packOptimally( bool preserveAABBTree )
{
    auto exp = packOptimally( preserveAABBTree, {} );
    assert( exp.has_value() );
    return std::move( *exp );
}

Expected<PackMapping> Mesh::packOptimally( bool preserveAABBTree, ProgressCallback cb )
{
    MR_TIMER;

    PackMapping map;
    AABBTreePointsOwner_.reset(); // points-tree will be invalidated anyway
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
        AABBTreeOwner_.update( [&map]( AABBTree& t ) { t.getLeafOrderAndReset( map.f ); } );
    }
    else
    {
        AABBTreeOwner_.reset();
        map.f = getOptimalFaceOrdering( *this );
    }
    if ( !reportProgress( cb, 0.3f ) )
        return unexpectedOperationCanceled();

    map.v = getVertexOrdering( map.f, topology );
    if ( !reportProgress( cb, 0.5f ) )
        return unexpectedOperationCanceled();

    map.e = getEdgeOrdering( map.f, topology );
    if ( !reportProgress( cb, 0.7f ) )
        return unexpectedOperationCanceled();

    if ( auto r = pack( map, subprogress( cb, 0.7f, 1.0f ) ); !r )
        return unexpected( std::move( r.error() ) );

    return map;
}

void Mesh::deleteFaces( const FaceBitSet & fs, const UndirectedEdgeBitSet * keepEdges )
{
    if ( fs.none() )
        return;
    topology.deleteFaces( fs, keepEdges );
    invalidateCaches(); // some points can be deleted as well
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

MeshProjectionResult Mesh::projectPoint( const Vector3f& point, float maxDistSq, const FaceBitSet * region, const AffineXf3f * xf ) const
{
    return findProjection( point, { *this, region }, maxDistSq, xf );
}

const AABBTree & Mesh::getAABBTree() const
{
    const auto & res = AABBTreeOwner_.getOrCreate( [this]{ return AABBTree( *this ); } );
    assert( res.numLeaves() == topology.numValidFaces() );
    return res;
}

const AABBTreePoints & Mesh::getAABBTreePoints() const
{
    const auto & res = AABBTreePointsOwner_.getOrCreate( [this]{ return AABBTreePoints( *this ); } );
    assert( res.orderedPoints().size() == topology.numValidVerts() );
    return res;
}

const Dipoles & Mesh::getDipoles() const
{
    if ( auto pRes = dipolesOwner_.get() )
        return *pRes; // fast path without tree access
    const auto & tree = getAABBTree(); // must be ready before lambda body for single-threaded Emscripten
    const auto & res = dipolesOwner_.getOrCreate(
        [this, &tree] {
            return calcDipoles( tree, *this );
        } );
    assert( res.size() == tree.nodes().size() );
    return res;
}

void Mesh::invalidateCaches( bool pointsChanged )
{
    AABBTreeOwner_.reset();
    if ( pointsChanged )
        AABBTreePointsOwner_.reset();
    dipolesOwner_.reset();
}

void Mesh::updateCaches( const VertBitSet & changedVerts )
{
    AABBTreeOwner_.update( [&]( AABBTree & tree )
    {
        assert( tree.numLeaves() == topology.numValidFaces() );
        tree.refit( *this, changedVerts );
    } );
    AABBTreePointsOwner_.update( [&]( AABBTreePoints & tree )
    {
        assert( tree.orderedPoints().size() == topology.numValidVerts() );
        tree.refit( points, changedVerts );
    } );
    dipolesOwner_.reset();
}

size_t Mesh::heapBytes() const
{
    return topology.heapBytes()
        + points.heapBytes()
        + AABBTreeOwner_.heapBytes()
        + AABBTreePointsOwner_.heapBytes()
        + dipolesOwner_.heapBytes();
}

void Mesh::shrinkToFit()
{
    MR_TIMER;
    topology.shrinkToFit();
    points.vec_.shrink_to_fit();
}

void Mesh::mirror( const Plane3f& plane )
{
    MR_TIMER;
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

TEST( MRMesh, isOutside )
{
    Mesh mesh = makeCube();
    EXPECT_TRUE( mesh.isOutside( Vector3f( 2, 0, 0 ) ) );
    EXPECT_FALSE( mesh.isOutside( Vector3f( 0, 0, 0 ) ) );
}

} //namespace MR
