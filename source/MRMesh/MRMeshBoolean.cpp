#include "MRMeshBoolean.h"
#include "MRBooleanOperation.h"
#include "MRMesh.h"
#include "MRMeshCollidePrecise.h"
#include "MRIntersectionContour.h"
#include "MRContoursCut.h"
#include "MRTimer.h"
#include "MRAffineXf3.h"
#include "MRLog.h"
#include "MRFillContour.h"
#include "MRPrecisePredicates3.h"
#include "MRRegionBoundary.h"
#include "MRMeshComponents.h"
#include "MRMeshCollide.h"
#include "MRMeshBuilder.h"
#include "MRParallelFor.h"
#include "MRBitSetParallelFor.h"
#include "MRBox.h"
#include "MRContoursStitch.h"
#include "MREdgePaths.h"
#include "MRRingIterator.h"
#include "MRPch/MRSpdlog.h"
#include "MRMeshFillHole.h"
#include "MRMeshFixer.h"
#include "MRAABBTree.h"
#include <limits>
#include <optional>

namespace MR
{

namespace
{

enum class EdgeTriComponent
{
    Edge,
    Tri,
};

Vector3f findEdgeTriIntersectionPoint( const Mesh& edgeMesh, EdgeId edge, const Mesh& triMesh, FaceId tri,
                                       const CoordinateConverters& converters,
                                       const AffineXf3f* rigidB2A = nullptr,
                                       EdgeTriComponent meshBComponent = EdgeTriComponent::Edge )
{
    Vector3f ev0, ev1;
    ev0 = edgeMesh.orgPnt( edge );
    ev1 = edgeMesh.destPnt( edge );

    Vector3f fv0, fv1, fv2;
    triMesh.getTriPoints( tri, fv0, fv1, fv2 );

    if ( rigidB2A )
    {
        const auto& xf = *rigidB2A;
        switch ( meshBComponent )
        {
        case EdgeTriComponent::Edge:
            ev0 = xf( ev0 );
            ev1 = xf( ev1 );
            break;
        case EdgeTriComponent::Tri:
            fv0 = xf( fv0 );
            fv1 = xf( fv1 );
            fv2 = xf( fv2 );
            break;
        }
    }

    return findTriangleSegmentIntersectionPrecise( fv0, fv1, fv2, ev0, ev1, converters );
}

void gatherEdgeInfo( const MeshTopology& topology, EdgeId e, FaceBitSet& faces, VertBitSet& orgs, VertBitSet& dests )
{
    if ( auto l = topology.left( e ) )
        faces.set( l );
    if ( auto r = topology.right( e ) )
        faces.set( r );
    orgs.set( topology.org( e ) );
    dests.set( topology.dest( e ) );
}

// removes the closed contours of two points, which have zero area, as their two segments coincide
void removeTwoPointLoops( ContinuousContours& contours )
{
    erase_if( contours, [] ( const ContinuousContour& c ) { return c.size() == 3 && isClosed( c ); } );
}

// cuts the surface along the given edges as cutAlongEdgeLoop does along a loop: every edge loses its left face to a new edge next to it,
// returned at the same index; then every vertex of the edges is split in its parts between the holes that this and earlier cuts marked in holes,
// one vertex per part: also where several loops cross, or where earlier stitches have separated the edges of a loop at a crossing
EdgePath cutAlongEdges( Mesh& mesh, const EdgePath& edges, EdgeBitSet& holes )
{
    auto& tp = mesh.topology;
    EdgePath res;
    res.reserve( edges.size() );
    for ( auto e : edges )
    {
        const auto t = tp.makeEdge();
        tp.splice( e, t );
        tp.splice( tp.prev( e.sym() ), t.sym() );
        res.push_back( t );
        // the holes start from e and t.sym() counterclockwise
        holes.autoResizeSet( e );
        holes.autoResizeSet( t.sym() );
    }
    // splits the ring of the origin of h, which starts a hole: every other start of a hole there takes the part from the end of the previous hole
    // to a new vertex, and h keeps the rest
    auto split = [&] ( EdgeId h )
    {
        for ( auto x = tp.next( h ); x != h; )
        {
            const auto next = tp.next( x );
            if ( !tp.left( x ) && holes.test( x ) )
            {
                const auto first = tp.next( h );
                tp.splice( h, x );
                tp.setOrg( first, tp.addVertId() );
                mesh.points.autoResizeSet( tp.org( first ), mesh.orgPnt( h ) );
            }
            x = next;
        }
    };
    for ( auto e : edges )
        split( e );
    for ( auto t : res )
        split( t.sym() );
    return res;
}

} //anonymous namespace

void convertIntFloatAllVerts( Mesh & mesh, const CoordinateConverters& conv )
{
    MR_TIMER;
    BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        mesh.points[v] = conv.toFloat( conv.toInt( mesh.points[v] ) );
    } );
}

BooleanResult booleanImpl( Mesh&& meshA, Mesh&& meshB, BooleanOperation operation, const BooleanParameters& params, BooleanInternalParameters intParams,
    bool copyOriginals = false );

BooleanResult boolean( const Mesh& meshA, const Mesh& meshB, BooleanOperation operation,
                       const AffineXf3f* rigidB2A /*= nullptr */, BooleanResultMapper* mapper /*= nullptr */, ProgressCallback cb )
{
    return boolean( meshA, meshB, operation, { .rigidB2A = rigidB2A, .mapper = mapper, .cb = cb } );
}

BooleanResult boolean( const Mesh& meshA, const Mesh& meshB, BooleanOperation operation, const BooleanParameters& params /*= {} */ )
{
    MR_TIMER;
    // the trees of the whole input meshes are constructed in booleanImpl and cached in them (never temporary trees of their parts),
    // which is important for many calls to Boolean for the same mesh to avoid tree construction on every call

    // the meshes are copied in booleanImpl only when their cuts are known, with the room the cuts need
    return booleanImpl( {}, {}, operation, params, { .originalMeshA = &meshA,.originalMeshB = &meshB }, true );
}

BooleanResult boolean( Mesh&& meshA, Mesh&& meshB, BooleanOperation operation, const BooleanParameters& params /*= {} */ )
{
    return booleanImpl( std::move( meshA ), std::move( meshB ), operation, params, {} );
}

namespace detail
{
struct JoinedSelfLoops
{
    int first{ 0 };
    int second{ 0 };
    int secondRot{ 0 };
};
Vector<JoinedSelfLoops> findSelfContoursMapping( const ContinuousContours& contours )
{
    assert( contours.size() % 2 == 0 );
    Vector<JoinedSelfLoops> holePairs( contours.size() / 2 );
    BitSet visitedConts( contours.size(), true );
    int i = 0;
    while ( visitedConts.any() )
    {
        size_t first = visitedConts.find_first();
        visitedConts.reset( first );
        auto fInter = contours[first][0];
        tbb::task_group_context ctx;
        ParallelFor( visitedConts.beginId(), visitedConts.endId(), [&] ( size_t next )
        {
            if ( ctx.is_group_execution_cancelled() || !visitedConts.test( next ) )
                return;
            int rot = 0;
            for ( const auto& nInter : contours[next] )
            {
                if ( ctx.is_group_execution_cancelled() )
                    return;
                ++rot;
                if ( fInter.isEdgeATriB() == nInter.isEdgeATriB() )
                    continue;
                if ( fInter.tri() != nInter.tri() )
                    continue;
                if ( fInter.edge != nInter.edge.sym() )
                    continue;
                if ( ctx.cancel_group_execution() )
                {
                    visitedConts.reset( next );
                    holePairs[i++] = { int( first ),int( next ),rot - 1 };
                }
                return;
            }
        } );
    }
    return holePairs;
}
}

BooleanResult forceBoolean( const Mesh& meshA, const Mesh& meshB, BooleanOperation operation, const BooleanParameters& inParams /*= {} */ )
{
    auto params = inParams;
    params.forceCut = true;
    auto res = boolean( meshA, meshB, operation, params );
    MeshBuilder::uniteCloseVertices( res.mesh, { .duplicateNonManifold = true } );
    FillHoleParams fhp;
    fhp.metric = getMinAreaMetric( res.mesh );
    fillHoles( res.mesh, res.mesh.topology.findHoleRepresentiveEdges(), fhp );
    return res;
}

Expected<MR::Mesh> selfBoolean( const Mesh& inMesh, FaceMap* outNew2Old )
{
    auto box = Box3d( inMesh.computeBoundingBox() );
    const auto converters = getNestedCoordinateConverters( box );
    Mesh mesh = inMesh;

    // find and order intersections
    auto intersections = findSelfCollidingEdgeTrisPrecise( mesh, converters.toInt );
    // two triangles with the same vertices glued along two edges (at a vertex of degree 2) or all three edges (a pillow) are a part without
    // volume, whose crossings with other parts are alike for both triangles, so that no cut tells them apart: such parts are eliminated
    // with their intersections, in one pass (where eliminating a pair makes its neighbours a pair, as in a stack of fins, and their
    // intersections came earlier, the new pair stays)
    for ( const auto& x : intersections )
        for ( auto f : { x.tri, mesh.topology.left( x.edge ) } )
            if ( f && mesh.topology.hasFace( f ) )
                for ( auto e : leftRing( mesh.topology, f ) )
                    if ( eliminateDoubleTris( mesh.topology, e, nullptr, true ) )
                        break;
    std::erase_if( intersections, [&] ( const EdgeTri& x ) { return !mesh.topology.hasFace( x.tri ) || mesh.topology.isLoneEdge( x.edge ); } );
    auto contours = orderSelfIntersectionContours( mesh.topology, intersections );
    if ( contours.empty() && !intersections.empty() )
        return unexpected( "Self-Boolean cannot order the intersections: the mesh has a face that is not a triangle" );
    removeTwoPointLoops( contours ); // both contours of a pair together, as they have the same points

    auto holePairs = detail::findSelfContoursMapping( contours );
    auto intContours = getOneMeshSelfIntersectionContours( mesh, contours, converters );
    // what the pairs below do with the contours: the first contour of a pair is closed through the vertices sv and fv and followed by
    // the second one, which is dropped then; or both contours of a pair are dropped
    struct PairChange
    {
        int second = -1; // set in the first contour of a merged pair
        VertId sv, fv;
        bool dropped = false;
    };
    Vector<PairChange> changes( contours.size() );

    // update non-closed
    ParallelFor( holePairs, [&] ( size_t hpId )
    {
        auto [f, s, r] = holePairs[hpId];
        auto& cf = contours[f];
        if ( isClosed( cf ) )
            return;

        if ( cf.capacity() < 2 * cf.size() + 3 )
            cf.reserve( 2 * cf.size() + 3 );

        auto& cs = contours[s];
        auto clearPair = [&]
        {
            cs.clear();
            cf.clear();
            changes[f].dropped = changes[s].dropped = true;
        };

        // close the contour through the third vertices of the faces where it starts and ends (a fold at a vertex shared by both sheets);
        // a contour that runs out to the mesh boundary has no such face, TODO: support this case
        if ( !mesh.topology.right( cf.front().edge ) || !mesh.topology.left( cf.back().edge ) )
        {
            clearPair();
            return;
        }

        auto prev = mesh.topology.prev( cf.front().edge );
        auto next = mesh.topology.next( cf.back().edge );
        auto svId = mesh.topology.dest( prev );
        auto fvId = mesh.topology.dest( next );

        auto triFVerts = mesh.topology.getTriVerts( cf.front().tri() );
        auto triBVerts = mesh.topology.getTriVerts( cf.back().tri() );
        if ( !( svId == triFVerts[0] || svId == triFVerts[1] || svId == triFVerts[2] ) ||
            !( fvId == triBVerts[0] || fvId == triBVerts[1] || fvId == triBVerts[2] ) )
        {
            // complex disjointed contour, TODO: support this case
            clearPair();
            return;
        }

        auto prevVET = cf.front();
        auto nextVET = cf.back();
        cf.insert( cf.begin(), prevVET );
        cf.insert( cf.end(), nextVET );
        cf.insert( cf.end(), std::make_move_iterator( cs.begin() ), std::make_move_iterator( cs.end() ) );
        cf.insert( cf.end(), prevVET );
        cs.clear();
        changes[f] = { .second = int( s ), .sv = svId, .fv = fvId };
        changes[s].dropped = true;
    } );

    if ( !holePairs.empty() )
    {
        // the intersections of the changed contours in a new array; the contours keep their indices
        OneMeshContours changed;
        auto& out = changed.intersections;
        out.reserve( intContours.intersections.size() + 3 * holePairs.size() );
        auto append = [&] ( const OneMeshContourView& c ) { out.insert( out.end(), c.intersections.begin(), c.intersections.end() ); };
        Vector<int> offsets( contours.size() + 1 ); // not grown in place: GCC sees push_back() to the default { 0 } as out of its bounds
        for ( int j = 0; j < contours.size(); ++j )
        {
            const auto& ch = changes[j];
            if ( ch.second >= 0 )
            {
                out.emplace_back( ch.sv, mesh.points[ch.sv] );
                append( intContours[j] );
                out.emplace_back( ch.fv, mesh.points[ch.fv] );
                append( intContours[ch.second] );
                out.emplace_back( ch.sv, mesh.points[ch.sv] );
            }
            else if ( !ch.dropped )
                append( intContours[j] );
            if ( ch.second >= 0 || intContours.closed.test( j ) )
                changed.closed.autoResizeSet( j );
            offsets[j + 1] = int( out.size() );
        }
        changed.offsets = std::move( offsets );
        intContours = std::move( changed );
    }

    // meshAVertsNum = 0 because both meshes are actually the same and no need to change the IDs of any mesh vertices
    const SortIntersectionsData sortData{ inMesh, contours, converters.toInt, nullptr, 0, false };
    const VertId firstNewVert( mesh.topology.vertSize() );
    const FaceId firstNewFace( mesh.topology.faceSize() );
    FaceMap new2Old;
    // where three sheets meet at a point, the contours cross on each of them: the cut splits them there, so the paths pass the crossings
    auto cutRes = cutMesh( mesh, intContours, { .sortData = &sortData, .new2OldMap = &new2Old,
        .forceFillMode = CutMeshParameters::ForceFill::SplitCrossings } );
    // the vertices at the crossings: new vertices that two paths leave (or one path twice), unlike the vertices at the points of the contours
    VertBitSet leftOnce( mesh.topology.vertSize() ), crossings( mesh.topology.vertSize() );
    for ( const auto& path : cutRes.resultCut )
        for ( auto e : path )
            if ( const auto v = mesh.topology.org( e ); v >= firstNewVert && leftOnce.test_set( v ) )
                crossings.set( v );
    if ( cutRes.fbsWithContourIntersections.any() )
    {
        // a bad face is filled well only where the cut split the crossings in it (the faces around such a crossing come from that face),
        // and all the faces made from it are triangles: where its monotone split failed, the planar fill can leave a face that is no
        // triangle (only the new faces can be such, as the old ones keep their sides), or the face is put back whole with its old id
        // next to the fills of its islands
        FaceBitSet split;
        for ( auto v : crossings )
            for ( auto e : orgRing( mesh.topology, v ) )
                if ( const auto f = mesh.topology.left( e ) )
                    split.autoResizeSet( new2Old[f] );
        auto bad = cutRes.fbsWithContourIntersections - ( split - mesh.topology.getValidFaces() );
        for ( auto f = firstNewFace; f < mesh.topology.faceSize(); ++f )
            if ( cutRes.fbsWithContourIntersections.test( new2Old[f] ) && !mesh.topology.isLeftTri( mesh.topology.edgeWithLeft( f ) ) )
                bad.set( new2Old[f] );
        if ( bad.any() )
            return unexpected( fmt::format( "Self-Boolean has nested intersections on {} faces", bad.count() ) );
    }
    convertIntFloatAllVerts( mesh, converters );

    // before any stitch changes the vertices: the second path of every closed pair goes along the first one from the same point
    // (r counts the points of the contour, while the path passes its crossings too), and a loop closed through fold vertices
    // goes from sv to fv on one sheet and back on the other one; the crossings must come at the same edges on both sheets,
    // so that the stitches join crossings with crossings and the points of the contour with their own pairs
    auto isCrossing = [&] ( EdgeId e ) { return crossings.test( mesh.topology.org( e ) ); };
    for ( auto [f, s, r] : holePairs )
    {
        if ( changes[f].dropped )
            continue;
        const auto& path = cutRes.resultCut[f];
        assert( isEdgeLoop( mesh.topology, path ) ); // until the cuts below split the crossings that it shares with other paths
        const auto n = path.size();
        bool aligned = false;
        if ( contours[s].empty() )
        {
            aligned = n % 2 == 0 && mesh.topology.org( path[n / 2] ) == changes[f].fv;
            for ( size_t i = 1; aligned && i < n / 2; ++i )
                aligned = isCrossing( path[i] ) == isCrossing( path[n - i] );
        }
        else
        {
            auto& leftLoopS = cutRes.resultCut[s];
            int j = 0;
            for ( int k = 0; k < r || isCrossing( leftLoopS[j] ); ++j )
                if ( !isCrossing( leftLoopS[j] ) )
                    ++k;
            std::rotate( leftLoopS.begin(), leftLoopS.begin() + j, leftLoopS.end() );
            reverse( leftLoopS );
            aligned = leftLoopS.size() == n;
            for ( size_t i = 1; aligned && i < n; ++i )
                aligned = isCrossing( path[i] ) == isCrossing( leftLoopS[i] );
        }
        if ( !aligned )
            return unexpected( "Self-Boolean has nested intersections split differently on the sheets" );
    }

    EdgeBitSet holes; // of the cuts below, until the stitches close them
    for ( auto [f, s, r] : holePairs )
    {
        if ( changes[f].dropped )
            continue;
        if ( !contours[s].empty() ) // isClosed( contours[f] )
        {
            const auto& leftLoopF = cutRes.resultCut[f];
            auto rightLoopF = cutAlongEdges( mesh, leftLoopF, holes );
            const auto& leftLoopS = cutRes.resultCut[s]; // rotated and reversed above to go along leftLoopF
            auto rightLoopS = cutAlongEdges( mesh, leftLoopS, holes );
            stitchContours( mesh.topology, leftLoopF, rightLoopS );
            stitchContours( mesh.topology, leftLoopS, rightLoopF );
        }
        else
        {
            auto& leftFirstLoop = cutRes.resultCut[f];
            auto rightFirstLoop = cutAlongEdges( mesh, leftFirstLoop, holes );
            EdgePath leftSecondPart( leftFirstLoop.begin() + leftFirstLoop.size() / 2, leftFirstLoop.end() );
            leftFirstLoop.resize( leftSecondPart.size() );
            reverse( leftSecondPart );
            
            EdgePath rightSecondPart( rightFirstLoop.begin() + rightFirstLoop.size() / 2, rightFirstLoop.end() );
            rightFirstLoop.resize( rightSecondPart.size() );
            reverse( rightSecondPart );

            stitchContours( mesh.topology, leftFirstLoop, leftSecondPart );
            stitchContours( mesh.topology, rightSecondPart, rightFirstLoop );
        }
    }

    if ( outNew2Old )
    {
        new2Old.resize( mesh.topology.faceSize() );
        *outNew2Old = std::move( new2Old );
    }
    return mesh;
}

Contours3f findIntersectionContours( const Mesh& meshA, const Mesh& meshB, const AffineXf3f* rigidB2A /*= nullptr */ )
{
    auto converters = getVectorConverters( meshA, meshB, rigidB2A );
    auto intersections = findCollidingEdgeTrisPrecise( meshA, meshB, converters.toInt, rigidB2A );
    auto contours = orderIntersectionContours( meshA.topology, meshB.topology, intersections );
    Contours3f res;
    getOneMeshIntersectionContours( meshA, meshB, contours, nullptr, nullptr, converters, rigidB2A, &res );
    return res;
}

/// finds the intersections of two meshes as findCollidingEdgeTrisPrecise does.
/// The tree of a whole mesh takes more time than the rest of Boolean, and only the faces near the other mesh can intersect it.
/// So if a mesh has no cached tree, the common part of the bounding boxes of the meshes is a small part of the box of this mesh,
/// and few of its faces are there, then a temporary tree of only these faces is used;
/// otherwise the tree of the whole mesh is taken or constructed and cached as usual.
/// \param allowPartialTrees false if the meshes belong to the caller, who can call Boolean for them again:
/// then the trees of the whole meshes are constructed and cached in them, so the next calls do not construct them
static PreciseCollisionResult findIntersections( const Mesh& a, const Mesh& b, const ConvertToIntVector& conv, const AffineXf3f* rigidB2A,
    bool allowPartialTrees )
{
    MR_TIMER;
    if ( !allowPartialTrees )
    {
        tbb::task_group taskGroup;
        taskGroup.run( [&] { a.getAABBTree(); } );
        b.getAABBTree();
        taskGroup.wait();
    }
    const AABBTree* aTree = a.getAABBTreeNotCreate();
    const AABBTree* bTree = b.getAABBTreeNotCreate();
    if ( aTree && bTree )
        return findCollidingEdgeTrisPrecise( a, *aTree, b, *bTree, conv, rigidB2A );

    // where the meshes can intersect, in the space of A; widened much more than the conversion in integers can move a point
    const Box3f boxA = aTree ? aTree->getBoundingBox() : a.computeBoundingBox();
    const Box3f boxB = b.computeBoundingBox( nullptr, rigidB2A );
    if ( !boxA.valid() || !boxB.valid() )
        return {};
    Box3f common = boxA.intersection( boxB );
    Box3f both = boxA;
    both.include( boxB );
    float maxAbs = 0;
    for ( int i = 0; i < 3; ++i )
        maxAbs = std::max( { maxAbs, std::abs( both.min[i] ), std::abs( both.max[i] ) } );
    const Vector3f margin = Vector3f::diagonal( 1e-5f * maxAbs + std::numeric_limits<float>::min() );
    common.min -= margin;
    common.max += margin;
    if ( !common.valid() )
        return {}; // the boxes are apart

    FaceBitSet aNear, bNear;
    std::optional<AABBTree> aLocal, bLocal;
    const auto findTree = [&common]( const Mesh& m, const Box3f& meshBox, const AffineXf3f* xf, FaceBitSet& nearFaces, std::optional<AABBTree>& local ) -> const AABBTree*
    {
        // if the common part takes much of the box of the mesh, then many of its faces are expected there,
        // and looking for them would only add a pass over the mesh before the construction of the whole tree
        const Vector3f meshSize = meshBox.size(), partSize = meshBox.intersection( common ).size();
        float partOfBox = 1;
        for ( int i = 0; i < 3; ++i )
            if ( meshSize[i] > 0 )
                partOfBox *= std::clamp( partSize[i] / meshSize[i], 0.0f, 1.0f );
        if ( partOfBox > 0.25f )
            return &m.getAABBTree();

        const auto& validFaces = m.topology.getValidFaces();
        nearFaces.resize( validFaces.size() );
        BitSetParallelFor( validFaces, [&]( FaceId f )
        {
            Box3f box;
            for ( VertId v : m.topology.getTriVerts( f ) )
                box.include( xf ? ( *xf )( m.points[v] ) : m.points[v] );
            if ( box.intersects( common ) )
                nearFaces.set( f );
        } );
        if ( nearFaces.count() * 4 > size_t( m.topology.numValidFaces() ) )
            return &m.getAABBTree();
        local.emplace( MeshPart{ m, &nearFaces } );
        return &*local;
    };
    tbb::task_group taskGroup;
    if ( !aTree )
        taskGroup.run( [&] { aTree = findTree( a, boxA, nullptr, aNear, aLocal ); } );
    if ( !bTree )
        bTree = findTree( b, boxB, rigidB2A, bNear, bLocal );
    taskGroup.wait();
    // a temporary tree comes with its faces as the region: an edge is checked once, from the triangle with the smaller index,
    // unless the other triangle is outside the region, so without the region an edge would be missed
    // if only its triangle with the greater index were in the tree
    return findCollidingEdgeTrisPrecise( { a, aLocal ? &aNear : nullptr }, *aTree, { b, bLocal ? &bNear : nullptr }, *bTree, conv, rigidB2A );
}

BooleanResult booleanImpl( Mesh&& meshA, Mesh&& meshB, BooleanOperation operation, const BooleanParameters& params, BooleanInternalParameters intParams,
    bool copyOriginals )
{
    MR_TIMER;
    // the meshes to read until they are cut: if copyOriginals, meshA and meshB are empty and get the copies of the original meshes just before their cuts
    const Mesh& srcA = copyOriginals ? *intParams.originalMeshA : meshA;
    const Mesh& srcB = copyOriginals ? *intParams.originalMeshB : meshB;
    // copies the original mesh with the capacity for its cut by these contours, so the cut does not reallocate the whole mesh
    // and if addContours, also with the room for all of the other mesh with its cut by them
    auto copyOriginal = [copyOriginals, &intParams] ( Mesh& to, const Mesh* from, const OneMeshContours* cutContours,
        const OneMeshContours* addContours = nullptr )
    {
        if ( !copyOriginals )
            return;
        MR_NAMED_TIMER( "copyOriginal" );
        if ( cutContours )
            reserveForCut( to, *from, *cutContours );
        if ( addContours )
        {
            const auto& add = intParams.originalMeshB->topology;
            const auto n = addContours->intersections.size();
            to.topology.vertReserve( to.topology.vertCapacity() + add.vertSize() + n );
            to.topology.edgeReserve( to.topology.edgeCapacity() + add.edgeSize() + 8 * n );
            to.topology.faceReserve( to.topology.faceCapacity() + add.faceSize() + 4 * n );
            to.points.reserve( to.points.capacity() + add.vertSize() + n );
        }
        to = *from;
    };
    BooleanResult result;
    CoordinateConverters converters;
    ContinuousContours contours;

    bool needCutMeshA = operation != BooleanOperation::InsideB && operation != BooleanOperation::OutsideB;
    bool needCutMeshB = operation != BooleanOperation::InsideA && operation != BooleanOperation::OutsideA;

    converters = getVectorConverters( srcA, srcB, params.rigidB2A );
    const size_t meshAVertsNum = srcA.topology.vertSize(); // before any cut of A

    // find and order intersections
    {
        // the original meshes are the caller's, the others are consumed here and their trees could not be reused
        const auto intersections = findIntersections( srcA, srcB, converters.toInt, params.rigidB2A, !copyOriginals );
        contours = orderIntersectionContours( srcA.topology, srcB.topology, intersections );
        if ( contours.empty() && !intersections.empty() )
            return { .errorString = "Boolean cannot order the intersections: a mesh has a face that is not a triangle" };
    }
    removeTwoPointLoops( contours );

    if ( params.cb && !params.cb( 0.0f ) )
        return { .errorString = stringOperationCanceled() };

    std::vector<EdgePath> cutA, cutB;
    OneMeshContours meshAContours;
    OneMeshContours meshBContours;
    // the meshes cut here, which need the sorting data: a precut output takes its mesh uncut
    const bool doCutA = needCutMeshA && !params.outPreCutA;
    const bool doCutB = needCutMeshB && !params.outPreCutB;
    // original mesh is needed to properly sort edges intersections while cutting meshes
    // as far as meshes is going to be change in the same time we need to take copies before start cutting
    Mesh meshACopyBuffer; // (only used if meshes were not copied before, otherwise old copy will be used)
    Mesh meshBCopyBuffer;// (only used if meshes were not copied before, otherwise old copy will be used)
    std::optional<SortIntersectionsData> dataForA;
    std::optional<SortIntersectionsData> dataForB;

    tbb::task_group taskGroup;
    if ( doCutA )
    {
        taskGroup.run( [&] ()
        {
            if ( doCutB )
            {
                if ( !intParams.originalMeshB )
                {
                    Timer t( "meshBCopyBuffer" );
                    meshBCopyBuffer = meshB;
                    intParams.originalMeshB = &meshBCopyBuffer;
                }
                dataForA.emplace( SortIntersectionsData{ *intParams.originalMeshB, contours, converters.toInt, params.rigidB2A, meshAVertsNum, false } );
            }
            else
                // B is stable so no need copy
                dataForA.emplace( SortIntersectionsData{ srcB, contours, converters.toInt, params.rigidB2A, meshAVertsNum, false } );
        } );
    }

    if ( doCutB )
    {
        if ( doCutA )
        {
            if ( !intParams.originalMeshA )
            {
                Timer t( "meshACopyBuffer" );
                meshACopyBuffer = meshA;
                intParams.originalMeshA = &meshACopyBuffer;
            }
            dataForB.emplace( SortIntersectionsData{ *intParams.originalMeshA, contours, converters.toInt, params.rigidB2A, meshAVertsNum, true } );
        }
        else
            // A is stable so no need to copy
            dataForB.emplace( SortIntersectionsData{ srcA, contours, converters.toInt, params.rigidB2A, meshAVertsNum, true } );
    }
    taskGroup.wait();

    if ( needCutMeshA || needCutMeshB )
        getOneMeshIntersectionContours( srcA, srcB, contours,
            needCutMeshA ? &meshAContours : nullptr,
            needCutMeshB ? &meshBContours : nullptr,
            converters, params.rigidB2A );

    if ( params.cb && !params.cb( 0.33f ) )
        return { .errorString = stringOperationCanceled() };

    bool needInsideA = operation == BooleanOperation::InsideA || operation == BooleanOperation::Intersection || operation == BooleanOperation::DifferenceBA;
    bool needInsideB = operation == BooleanOperation::InsideB || operation == BooleanOperation::Intersection || operation == BooleanOperation::DifferenceAB;
    // doBooleanOperation adds the part of cut mesh B to cut mesh A: the room for all of it is reserved in the copy of A,
    // so that the addition does not reallocate and copy the whole of A
    const bool reserveBinA = copyOriginals && doCutA && doCutB && !params.outPreCutA;
    if ( doCutA )
    {
        taskGroup.run( [&] ()
        {
            copyOriginal( meshA, intParams.originalMeshA, &meshAContours, reserveBinA ? &meshBContours : nullptr );
            Timer t( "CutMeshA" );
            FaceMap* cut2oldAPtr = params.mapper ? &params.mapper->maps[int( BooleanResultMapper::MapObject::A )].cut2origin : nullptr;
            // cut meshes
            CutMeshParameters cmParams;
            cmParams.sortData = &*dataForA;
            cmParams.new2OldMap = cut2oldAPtr;
            if ( params.forceCut )
                cmParams.forceFillMode = CutMeshParameters::ForceFill::All;
            else
                cmParams.fillPart = needInsideA ? CutMeshParameters::FillPart::Left : CutMeshParameters::FillPart::Right;
            auto res = cutMesh( meshA, meshAContours, cmParams );
            meshAContours = {}; // free memory
            result.meshABadContourFaces = std::move( res.fbsWithContourIntersections );
            cutA = std::move( res.resultCut );
        } );
    }

    if ( doCutB )
    {
        copyOriginal( meshB, intParams.originalMeshB, &meshBContours );
        Timer t( "CutMeshB" );
        FaceMap* cut2oldBPtr = params.mapper ? &params.mapper->maps[int( BooleanResultMapper::MapObject::B )].cut2origin : nullptr;
        // cut meshes
        CutMeshParameters cmParams;
        cmParams.sortData = &*dataForB;
        cmParams.new2OldMap = cut2oldBPtr;
        if ( params.forceCut )
            cmParams.forceFillMode = CutMeshParameters::ForceFill::All;
        else
            cmParams.fillPart = needInsideB ? CutMeshParameters::FillPart::Right : CutMeshParameters::FillPart::Left;
        auto res = cutMesh( meshB, meshBContours, cmParams );
        meshBContours = {}; // free memory
        result.meshBBadContourFaces = std::move( res.fbsWithContourIntersections );
        cutB = std::move( res.resultCut );
    }
    taskGroup.wait();
    if ( !doCutA )
        copyOriginal( meshA, intParams.originalMeshA, nullptr );
    if ( !doCutB )
        copyOriginal( meshB, intParams.originalMeshB, nullptr );

    // the precut outputs take the meshes only after the cuts: where one mesh is not cut (by the operation or for its precut output),
    // the cut of the other one sorts its intersections by that mesh itself, not by a copy;
    // nothing reads the intersections after that, so the last output takes them
    auto fillPreCut = [&] ( BooleanPreCutResult& out, Mesh& mesh, OneMeshContours& meshContours, bool isMeshA, bool last )
    {
        out.mesh = std::move( mesh );
        out.contours = std::move( meshContours );
        out.intersections = last ? std::move( contours ) : contours;
        out.toInt = converters.toInt;
        out.rigidB2A = params.rigidB2A ? std::optional<AffineXf3f>( *params.rigidB2A ) : std::nullopt;
        out.meshAVertsNum = meshAVertsNum;
        out.isMeshA = isMeshA;
    };
    if ( params.outPreCutA )
        fillPreCut( *params.outPreCutA, meshA, meshAContours, true, !params.outPreCutB );
    if ( params.outPreCutB )
        fillPreCut( *params.outPreCutB, meshB, meshBContours, false, true );

    if ( !params.forceCut && result.meshABadContourFaces.any() )
    {
        result.errorString = "Bad contour on " + std::to_string( result.meshABadContourFaces.count() ) + " mesh A faces, " +
            "probably mesh B has self-intersections on contours lying on these faces.";
        return result;
    }
    else if ( !params.forceCut && result.meshBBadContourFaces.any() )
    {
        result.errorString = "Bad contour on " + std::to_string( result.meshBBadContourFaces.count() ) + " mesh B faces, " +
            "probably mesh A has self-intersections on contours lying on these faces.";
        return result;
    }


    if ( params.cb && !params.cb( 0.66f ) )
        return { .errorString = stringOperationCanceled() };

    if ( params.outPreCutA || params.outPreCutB )
        return {};

    intParams.optionalOutCut = params.outCutEdges;
    intParams.graphCutSeparation = params.forceCut;
    intParams.converters = &converters;
    // do operation
    auto res = doBooleanOperation( std::move( meshA ), std::move( meshB ), std::move( cutA ), std::move( cutB ), 
        operation, params.rigidB2A, params.mapper, params.mergeAllNonIntersectingComponents, intParams );

    if ( params.cb && !params.cb( 1.0f ) )
        return { .errorString = stringOperationCanceled() };

    if ( res.has_value() )
    {
        result.mesh = std::move( res.value() );
        convertIntFloatAllVerts( result.mesh, converters );
    }
    else
        result.errorString = res.error();
    return result;
}

Expected<BooleanResultPoints> getBooleanPoints( const Mesh& meshA, const Mesh& meshB,
    BooleanOperation operation, const AffineXf3f* rigidB2A )
{
    MR_TIMER;

    BooleanResultPoints result;
    result.meshAVerts.resize( meshA.topology.lastValidVert() + 1 );
    result.meshBVerts.resize( meshB.topology.lastValidVert() + 1 );

    const auto converters = getVectorConverters( meshA, meshB, rigidB2A );
    const auto intersections = findCollidingEdgeTrisPrecise( meshA, meshB, converters.toInt, rigidB2A );
    result.intersectionPoints.reserve( intersections.size() );

    FaceBitSet collFacesA, collFacesB;
    VertBitSet destVertsA, destVertsB, orgVertsA, orgVertsB;
    collFacesA.resize( meshA.topology.lastValidFace() + 1 );
    collFacesB.resize( meshB.topology.lastValidFace() + 1 );
    orgVertsA.resize( meshA.topology.lastValidVert() + 1 );
    orgVertsB.resize( meshB.topology.lastValidVert() + 1 );
    destVertsA.resize( meshA.topology.lastValidVert() + 1 );
    destVertsB.resize( meshB.topology.lastValidVert() + 1 );
    for ( const auto& et : intersections )
    {
        if ( et.isEdgeATriB() )
        {
            gatherEdgeInfo( meshA.topology, et.edge, collFacesA, orgVertsA, destVertsA );
            collFacesB.set( et.tri() );

            const auto isect = findEdgeTriIntersectionPoint( meshA, et.edge, meshB, et.tri(), converters, rigidB2A, EdgeTriComponent::Tri );
            result.intersectionPoints.emplace_back( isect );
        }
        else
        {
            gatherEdgeInfo( meshB.topology, et.edge, collFacesB, orgVertsB, destVertsB );
            collFacesA.set( et.tri() );

            const auto isect = findEdgeTriIntersectionPoint( meshB, et.edge, meshA, et.tri(), converters, rigidB2A, EdgeTriComponent::Edge );
            result.intersectionPoints.emplace_back( isect );
        }
    }

    if ( orgVertsA.intersects( destVertsA ) || orgVertsB.intersects( destVertsB ) )
    {
        // in this case we are not able to detect inside outside correctly
        BooleanResultMapper mapper;
        auto boolRes = MR::boolean( meshA, meshB, operation, rigidB2A, &mapper );
        if ( !boolRes.valid() )
            return unexpected( boolRes.errorString );

        if ( !mapper.maps[int( BooleanResultMapper::MapObject::A )].old2newVerts.empty() )
            for ( auto v : meshA.topology.getValidVerts() )
            {
                auto vn = mapper.maps[int( BooleanResultMapper::MapObject::A )].old2newVerts[v];
                if ( vn.valid() )
                    result.meshAVerts.set( v );
            }

        if ( !mapper.maps[int( BooleanResultMapper::MapObject::B )].old2newVerts.empty() )
            for ( auto v : meshB.topology.getValidVerts() )
            {
                auto vn = mapper.maps[int( BooleanResultMapper::MapObject::B )].old2newVerts[v];
                if ( vn.valid() )
                    result.meshBVerts.set( v );
            }

        return result;
    }

    auto collBordersA = findLeftBoundary( meshA.topology, collFacesA );
    auto collBordersB = findLeftBoundary( meshB.topology, collFacesB );

    const bool needInsidePartA = ( operation == BooleanOperation::Intersection || operation == BooleanOperation::InsideA || operation == BooleanOperation::DifferenceBA );
    const bool needInsidePartB = ( operation == BooleanOperation::Intersection || operation == BooleanOperation::InsideB || operation == BooleanOperation::DifferenceAB );

    if ( operation != BooleanOperation::InsideB && operation != BooleanOperation::OutsideB )
    {
        std::erase_if( collBordersA, [&] ( const EdgeLoop& edgeLoop )
        {
            return needInsidePartA != destVertsA.test( meshA.topology.dest( edgeLoop.front() ) );
        } );

        collFacesA = fillContourLeft( meshA.topology, collBordersA );
        result.meshAVerts = getInnerVerts( meshA.topology, collFacesA );

        // reset outer
        result.meshAVerts -= ( needInsidePartA ? destVertsA : orgVertsA );

        const auto aComponents = MeshComponents::getAllComponents(meshA);

        for ( const auto& aComponent : aComponents )
        {
            const auto aComponentVerts = getInnerVerts( meshA.topology, aComponent );
            if ( aComponentVerts.intersects( result.meshAVerts ) )
                continue;
            const bool inside = isInside( MeshPart( meshA, &aComponent ), MeshPart( meshB ), rigidB2A );
            if ( needInsidePartA == inside )
            {
                result.meshAVerts |= aComponentVerts;
            }
        }
    }

    if ( operation != BooleanOperation::InsideA && operation != BooleanOperation::OutsideA )
    {
        std::erase_if( collBordersB, [&] ( const EdgeLoop& edgeLoop )
        {
            return needInsidePartB != destVertsB.test( meshB.topology.dest( edgeLoop.front() ) );
        } );

        collFacesB = fillContourLeft(meshB.topology, collBordersB);
        result.meshBVerts = getInnerVerts( meshB.topology, collFacesB );

        // reset outer
        result.meshBVerts -= ( needInsidePartB ? destVertsB : orgVertsB );

        const auto bComponents = MeshComponents::getAllComponents(meshB);
        std::optional<AffineXf3f> rigidA2B;
        if ( rigidB2A )
            rigidA2B = rigidB2A->inverse();

        for ( const auto& bComponent : bComponents )
        {
            const auto bComponentVerts = getInnerVerts( meshB.topology, bComponent );
            if ( bComponentVerts.intersects( result.meshBVerts ) )
                continue;

            const bool inside = isInside( MeshPart( meshB, &bComponent ), MeshPart( meshA ), rigidA2B ? &*rigidA2B : nullptr );

            if ( needInsidePartB == inside  )
            {
                result.meshBVerts |= bComponentVerts;
            }
        }
    }

    return result;
}

} //namespace MR
