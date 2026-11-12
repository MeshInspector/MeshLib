#include "MRMeshCollidePrecise.h"
#include "MRAABBTreeTraversal.hpp"
#include "MRMesh.h"
#include "MRPrecisePredicates3.h"
#include "MRFaceFace.h"
#include "MRTimer.h"
#include "MRParallelFor.h"
#include "MRRingIterator.h"
#include "MRBuffer.h"
#include <array>
#include <atomic>
#include <optional>

namespace MR
{

namespace
{

/// whether the int boxes of two float boxes intersect (int boxes are used for consistency with precise intersections), given minGap = 2 / conv.invRange:
/// the conversion is monotone, so the int boxes intersect if the float boxes do, and do not if the float boxes are at least two int units apart;
/// only the boxes in between are converted
bool preciseBoxesIntersect( const Box3f & a, const Box3f & b, const ConvertToIntVector & conv, double minGap )
{
    // all comparisons as <=, which fail on NaN
    if ( a.min.x <= b.max.x && b.min.x <= a.max.x
      && a.min.y <= b.max.y && b.min.y <= a.max.y
      && a.min.z <= b.max.z && b.min.z <= a.max.z )
        return true;
    for ( int i = 0; i < 3; ++i )
        if ( double( a.min[i] ) - b.max[i] >= minGap || double( b.min[i] ) - a.max[i] >= minGap )
            return false;
    return Box3i{ conv( a.min ), conv( a.max ) }.intersects( Box3i{ conv( b.min ), conv( b.max ) } );
}

/// decides by preciseBoxesIntersect whether the boxes of nodes or triangles of tree A and of tree B after the transformation xf (if given) intersect
class PreciseBoxesIntersection
{
public:
    PreciseBoxesIntersection( const AABBTree & aTree, const AABBTree & bTree, const ConvertToIntVector & conv, const AffineXf3f * xf )
        : aTree_( aTree ), bTree_( bTree ), conv_( conv ), minGap_( 2 / conv.invRange )
    {
        if ( xf )
        {
            bNodeBoxes_.emplace( bTree.nodes().size(), *xf );
            bLeafBoxes_.emplace( bTree.leafBoxes().size(), *xf );
        }
    }

    bool nodes( NodeId a, NodeId b ) const { return test_( aTree_.nodes()[a].box, bNodeBox_( b ) ); }
    bool triNode( int a, NodeId b ) const { return test_( aTree_.leafBoxes()[a], bNodeBox_( b ) ); }
    bool tris( int a, int b ) const { return test_( aTree_.leafBoxes()[a], bLeafBoxes_ ? bLeafBoxes_->get( b, bTree_.leafBoxes()[b] ) : bTree_.leafBoxes()[b] ); }

private:
    bool test_( const Box3f & a, const Box3f & b ) const { return preciseBoxesIntersect( a, b, conv_, minGap_ ); }
    Box3f bNodeBox_( NodeId b ) const { return bNodeBoxes_ ? bNodeBoxes_->get( b, bTree_.nodes()[b].box ) : bTree_.nodes()[b].box; }

    const AABBTree & aTree_;
    const AABBTree & bTree_;
    const ConvertToIntVector & conv_;
    double minGap_ = 0;
    std::optional<LazyTransformedBoxes<NodeId>> bNodeBoxes_;
    std::optional<LazyTransformedBoxes<size_t>> bLeafBoxes_;
};

/// finds all pairs of colliding edges and triangles of two mesh parts, given the AABB trees with (at least) all triangles of the parts
PreciseCollisionResult findCollidingEdgeTrisPrecise( const MeshPart & a, const AABBTree & aTree,
    const MeshPart & b, const AABBTree & bTree, const PreciseBoxesIntersection & boxesIntersect,
    ConvertToIntVector conv, const AffineXf3f * rigidB2A, bool anyIntersection )
{
    PreciseCollisionResult res;
    if ( aTree.nodes().empty() || bTree.nodes().empty() )
        return res;

    Timer t( "2 top subtasks" );
    if ( !boxesIntersect.nodes( NodeId{ 0 }, NodeId{ 0 } ) )
        return res;
    auto nodesIntersect = [&boxesIntersect]( NodeId x, NodeId y ) { return boxesIntersect.nodes( x, y ); };
    auto split = [&]( const WideNodePair & s, auto && push ) { return splitWidePair( aTree, bTree, s, nodesIntersect, push ); };
    // 7 levels -> at most 4^7 = 2^14 subtasks
    const auto subtasks = makeWideSubtasks( { { NodeId{ 0 }, NodeId{ 0 } } }, 7, split );

    // we do not check an edge if its right triangle has smaller index and also in the mesh part
    auto checkEdge = [&]( EdgeId e, const MeshPart & mp )
    {
        const auto r = mp.mesh.topology.right( e );
        if ( !r )
            return true;
        if ( mp.region && !mp.region->test( r ) )
            return true;

        const auto l = mp.mesh.topology.left( e );
        assert ( l );
        return l < r;
    };

    // what the precise tests need to know about a triangle, found once for all its pairs with the triangles of another leaf
    struct TriData
    {
        PreciseVertCoords vc[3];
        EdgeId edges[3]; ///< edges[k] from vc[k] to vc[k+1] has the triangle at the left, invalid if the edge is not checked
    };
    struct LeafData
    {
        NodeId node;
        int ready = 0; ///< bit k is set if tris[k] is filled
        TriData tris[AABBTree::MaxLeafTris];
    };
    const int aVertsSize = (int)a.mesh.topology.vertSize();
    auto fillTri = [&]( TriData & td, const MeshPart & mp, const AABBTree::LeafTri & lt, bool isB )
    {
        for ( int j = 0; j < 3; ++j )
        {
            const auto v = lt.verts[j];
            const auto pt = mp.mesh.points[v];
            td.vc[j].pt = conv( isB && rigidB2A ? (*rigidB2A)( pt ) : pt );
            td.vc[j].id = isB ? v + aVertsSize : v;
        }
        EdgeId e = mp.mesh.topology.edgeWithLeft( lt.face );
        for ( int j = 0; j < 3; ++j )
        {
            td.edges[j] = checkEdge( e, mp ) ? e : EdgeId{};
            e = mp.mesh.topology.prev( e.sym() );
        }
    };
    // returns the data of triangle i (in leafTris()) of the leaf, which data are kept in ld
    auto getTri = [&]( LeafData & ld, const MeshPart & mp, const AABBTree & tree, NodeId leaf, int i, bool isB ) -> const TriData &
    {
        if ( ld.node != leaf )
        {
            ld.node = leaf;
            ld.ready = 0;
        }
        const int k = i - tree.nodes()[leaf].first;
        auto & td = ld.tris[k];
        if ( !( ld.ready & ( 1 << k ) ) )
        {
            fillTri( td, mp, tree.leafTris()[i], isB );
            ld.ready |= 1 << k;
        }
        return td;
    };
    auto checkTwoTris = [&]( const TriData & at, FaceId aTri, const TriData & bt, FaceId bTri, PreciseCollisionResult & res )
    {
        // check edges from A
        for ( int k = 0; k < 3; ++k )
        {
            const auto e = at.edges[k];
            if ( !e )
                continue;
            if ( auto isect = doTriangleSegmentIntersect( { bt.vc[0], bt.vc[1], bt.vc[2], at.vc[k], at.vc[k < 2 ? k + 1 : 0] } ) )
                res.emplace_back( true, isect.dIsLeftFromABC ? e : e.sym(), bTri );
        }
        // check edges from B
        for ( int k = 0; k < 3; ++k )
        {
            const auto e = bt.edges[k];
            if ( !e )
                continue;
            if ( auto isect = doTriangleSegmentIntersect( { at.vc[0], at.vc[1], at.vc[2], bt.vc[k], bt.vc[k < 2 ? k + 1 : 0] } ) )
                res.emplace_back( false, isect.dIsLeftFromABC ? e : e.sym(), aTri );
        }
    };

    // checks subtasks in parallel
    t.restart( "3 process" );

    struct ThreadData
    {
        PreciseCollisionResult res;
        Vector<WideNodePair> subtasks;
        LeafData aLeaf, bLeaf;
    };

    tbb::enumerable_thread_specific<ThreadData> threadData;

    struct SubtaskRes
    {
        PreciseCollisionResult * vec = nullptr;
        int first = 0;
        int last = 0;
    };

    Vector<SubtaskRes> subtaskRes( subtasks.size() );

    std::atomic<bool> anyIntersectionAtm{ false };
    ParallelFor( subtasks, threadData, [&]( size_t is, ThreadData & tls )
    {
        Vector<WideNodePair>& mySubtasks = tls.subtasks;
        assert( mySubtasks.empty() );
        mySubtasks.push_back( subtasks[is] );
        SubtaskRes myRes{ .vec = &tls.res };
        myRes.first = (int)myRes.vec->size();
        while ( !mySubtasks.empty() )
        {
            if ( anyIntersection && anyIntersectionAtm.load( std::memory_order_relaxed ) )
                break;
            const auto s = mySubtasks.back();
            mySubtasks.pop_back();
            if ( split( s, [&mySubtasks]( const WideNodePair & c ) { mySubtasks.push_back( c ); } ) )
                continue;
            const bool goOn = forWideLeafTris( aTree, bTree, s,
                [&boxesIntersect]( int i, NodeId n ) { return boxesIntersect.triNode( i, n ); },
                [&boxesIntersect]( int i, int j ) { return boxesIntersect.tris( i, j ); },
                [&]( int i, int j )
                {
                    const auto & at = aTree.leafTris()[i];
                    if ( a.region && !a.region->test( at.face ) )
                        return true;
                    const auto & bt = bTree.leafTris()[j];
                    if ( b.region && !b.region->test( bt.face ) )
                        return true;
                    checkTwoTris( getTri( tls.aLeaf, a, aTree, s.a, i, false ), at.face, getTri( tls.bLeaf, b, bTree, s.b, j, true ), bt.face, *myRes.vec );
                    return !anyIntersection || myRes.vec->empty();
                } );
            if ( !goOn )
            {
                anyIntersectionAtm.store( true, std::memory_order_relaxed );
                break;
            }
        }
        mySubtasks.clear();
        myRes.last = (int)myRes.vec->size();
        subtaskRes[is] = std::move( myRes );
    } );

    // unite results from sub-trees into final vector
    t.restart( "4 unite" );
    size_t cols = 0;
    for ( const auto & s : subtaskRes )
        cols += s.last - s.first;
    res.reserve( cols );
    for ( const auto & s : subtaskRes )
        res.insert( res.end(), s.vec->begin() + s.first, s.vec->begin() + s.last );

    return res;
}

} //anonymous namespace

PreciseCollisionResult findCollidingEdgeTrisPrecise( const MeshPart & a, const MeshPart & b,
    ConvertToIntVector conv, const AffineXf3f * rigidB2A, bool anyIntersection )
{
    MR_TIMER;
    const AABBTree & aTree = a.mesh.getAABBTree();
    const AABBTree & bTree = b.mesh.getAABBTree();
    const PreciseBoxesIntersection boxesIntersect( aTree, bTree, conv, rigidB2A );
    return findCollidingEdgeTrisPrecise( a, aTree, b, bTree, boxesIntersect, conv, rigidB2A, anyIntersection );
}

PreciseCollisionResult findCollidingEdgeTrisPrecise( const MeshPart & a, const AABBTree & aTree, const MeshPart & b, const AABBTree & bTree,
    ConvertToIntVector conv, const AffineXf3f * rigidB2A, bool anyIntersection )
{
    MR_TIMER;
    if ( aTree.nodes().empty() || bTree.nodes().empty() )
        return {};
    const PreciseBoxesIntersection boxesIntersect( aTree, bTree, conv, rigidB2A );
    return findCollidingEdgeTrisPrecise( a, aTree, b, bTree, boxesIntersect, conv, rigidB2A, anyIntersection );
}
std::vector<EdgeTri> findCollidingEdgeTrisPrecise( 
    const Mesh & a, const std::vector<EdgeId> & edgesA,
    const Mesh & b, const std::vector<FaceId> & facesB,
    ConvertToIntVector conv, const AffineXf3f * rigidB2A )
{
    using EdgePreciseCoords = std::array<PreciseVertCoords, 2>;
    std::vector<EdgePreciseCoords> edgeACoords( edgesA.size() );
    for ( int i = 0; i < edgesA.size(); ++i )
    {
        EdgeId eA = edgesA[i];
        auto & avc = edgeACoords[i];
        avc[0].id = a.topology.org( eA );
        avc[0].pt = conv( a.points[avc[0].id] );
        avc[1].id = a.topology.dest( eA );
        avc[1].pt = conv( a.points[avc[1].id] );
    }

    const int aVertsSize = (int)a.topology.vertSize();
    using TriPreciseCoords = std::array<PreciseVertCoords, 3>;
    std::vector<TriPreciseCoords> faceBCoords( facesB.size() );
    for ( int i = 0; i < facesB.size(); ++i )
    {
        FaceId fB = facesB[i];
        auto & bvc = faceBCoords[i];
        b.topology.getTriVerts( fB, bvc[0].id, bvc[1].id, bvc[2].id );

        for ( int j = 0; j < 3; ++j )
        {
            const auto bf = b.points[bvc[j].id];
            bvc[j].pt = conv( rigidB2A ? (*rigidB2A)( bf ) : bf );
            bvc[j].id += aVertsSize;
        }
    }

    std::vector<EdgeTri> res;
    for ( int i = 0; i < edgesA.size(); ++i )
    {
        EdgeId eA = edgesA[i];
        const auto & avc = edgeACoords[i];
        for ( int j = 0; j < facesB.size(); ++j )
        {
            FaceId fB = facesB[j];
            const auto & bvc = faceBCoords[j];
            auto isect = doTriangleSegmentIntersect( { bvc[0], bvc[1], bvc[2], avc[0], avc[1] } );
            if ( !isect )
                continue;
            res.emplace_back( isect.dIsLeftFromABC ? eA : eA.sym(), fB );
        }
    }

    return res;
}

std::vector<EdgeTri> findCollidingEdgeTrisPrecise(
    const Mesh & a, const std::vector<FaceId> & facesA,
    const Mesh & b, const std::vector<EdgeId> & edgesB,
    ConvertToIntVector conv, const AffineXf3f * rigidB2A )
{
    using TriPreciseCoords = std::array<PreciseVertCoords, 3>;
    std::vector<TriPreciseCoords> faceACoords( facesA.size() );
    for ( int i = 0; i < facesA.size(); ++i )
    {
        FaceId fA = facesA[i];
        auto & avc = faceACoords[i];
        a.topology.getTriVerts( fA, avc[0].id, avc[1].id, avc[2].id );
        for ( int j = 0; j < 3; ++j )
        {
            const auto af = a.points[avc[j].id];
            avc[j].pt = conv( af );
        }
    }

    const int aVertsSize = (int)a.topology.vertSize();
    using EdgePreciseCoords = std::array<PreciseVertCoords, 2>;
    std::vector<EdgePreciseCoords> edgeBCoords( edgesB.size() );
    for ( int i = 0; i < edgesB.size(); ++i )
    {
        EdgeId eB = edgesB[i];
        auto & bvc = edgeBCoords[i];
        bvc[0].id = b.topology.org( eB );
        bvc[1].id = b.topology.dest( eB );
        for ( int j = 0; j < 2; ++j )
        {
            const auto bf = b.points[bvc[j].id];
            bvc[j].pt = conv( rigidB2A ? (*rigidB2A)( bf ) : bf );
            bvc[j].id += aVertsSize;
        }
    }

    std::vector<EdgeTri> res;
    for ( int i = 0; i < facesA.size(); ++i )
    {
        FaceId fA = facesA[i];
        const auto & avc = faceACoords[i];
        for ( int j = 0; j < edgesB.size(); ++j )
        {
            EdgeId eB = edgesB[j];
            const auto & bvc = edgeBCoords[j];
            auto isect = doTriangleSegmentIntersect( { avc[0], avc[1], avc[2], bvc[0], bvc[1] } );
            if ( !isect )
                continue;
            res.emplace_back( isect.dIsLeftFromABC ? eB : eB.sym(), fA );
        }
    }

    return res;
}

void updateCollidingEdgeTrisPrecise( PreciseCollisionResult & res,
    const Mesh & a, VertId aFirstNewVert, const Mesh & b, VertId bFirstNewVert,
    ConvertToIntVector conv, const AffineXf3f * rigidB2A )
{
    MR_TIMER;
    // if many faces were split, a new search is faster: e.g. on two 3366-vertex spheres the update is faster with 1/26 of the vertices new,
    // and 1.3 times slower with 1/9 of them new; in Boolean only the faces with lone contours are split, usually much fewer
    const size_t numVerts = a.topology.vertSize() + b.topology.vertSize();
    if ( 32 * ( numVerts - size_t( aFirstNewVert ) - size_t( bFirstNewVert ) ) > numVerts )
    {
        res = findCollidingEdgeTrisPrecise( a, b, conv, rigidB2A );
        return;
    }

    // first the triangles of A changed by the splits, then the ones of B;
    // if both meshes were split, the pass of B removes and finds again the few intersections of the pass of A
    // between the changed triangles of A and B and their edges
    for ( bool inA : { true, false } )
    {
        const Mesh & mesh = inA ? a : b;

        // the triangles around new vertices: the split ones (keeping their ids) and the new ones
        FaceBitSet tris( mesh.topology.faceSize() );
        for ( VertId v = inA ? aFirstNewVert : bFirstNewVert; v < mesh.topology.vertSize(); ++v )
            for ( auto e : orgRing( mesh.topology, v ) )
                if ( auto t = mesh.topology.left( e ) )
                    tris.set( t );
        if ( tris.none() )
            continue;

        // remove all intersections with these triangles and their edges
        UndirectedEdgeBitSet edges( mesh.topology.undirectedEdgeSize() );
        for ( auto t : tris )
            for ( auto e : leftRing( mesh.topology, t ) )
                edges.set( e.undirected() );
        erase_if( res, [&]( const VarEdgeTri & et )
        {
            // both tests without a branch on the kind of the record, which alternates unpredictably:
            // the test of the ids of the other mesh is masked out
            const bool byEdge = et.isEdgeATriB() == inA;
            const bool edgeHit = edges.test( et.edge.undirected() );
            const bool triHit = tris.test( et.tri() );
            return ( byEdge & edgeHit ) | ( !byEdge & triHit );
        } );

        // and find them again: the intersections of these triangles and their edges with the whole other mesh,
        // the same as findCollidingEdgeTrisPrecise( MeshPart{ mesh, &tris }, other ) finds, but with the tree of these triangles only
        const MeshPart part{ mesh, &tris };
        const AABBTree trisTree( part );
        const auto & otherTree = ( inA ? b : a ).getAABBTree();
        const PreciseBoxesIntersection boxesIntersect( inA ? trisTree : otherTree, inA ? otherTree : trisTree, conv, rigidB2A );
        const auto found = inA ? findCollidingEdgeTrisPrecise( part, trisTree, b, otherTree, boxesIntersect, conv, rigidB2A, false )
            : findCollidingEdgeTrisPrecise( a, otherTree, part, trisTree, boxesIntersect, conv, rigidB2A, false );
        res.insert( res.end(), found.begin(), found.end() );
    }
}


std::vector<EdgeTri> findSelfCollidingEdgeTrisPrecise( const MeshPart& mp, ConvertToIntVector conv, bool anyIntersection /*= false */, 
    const AffineXf3f* rigidB2A, int aVertsSize )
{
    MR_TIMER;

    std::vector<EdgeTri> res;
    const AABBTree& tree = mp.mesh.getAABBTree();
    if ( tree.nodes().empty() )
        return res;

    // the traversal visits every node, and each many times, so the boxes after the transformation are computed once in advance
    Vector<Box3f, NodeId> xfNodeBoxes;
    Vector<Box3f> xfLeafBoxes;
    if ( rigidB2A )
    {
        xfNodeBoxes.resizeNoInit( tree.nodes().size() );
        ParallelFor( xfNodeBoxes, [&]( NodeId i ) { xfNodeBoxes[i] = transformed( tree.nodes()[i].box, *rigidB2A ); } );
        xfLeafBoxes.resizeNoInit( tree.leafBoxes().size() );
        ParallelFor( xfLeafBoxes, [&]( size_t i ) { xfLeafBoxes[i] = transformed( tree.leafBoxes()[i], *rigidB2A ); } );
    }
    const double minGap = 2 / conv.invRange;
    auto nodeBox = [&]( NodeId n ) -> const Box3f & { return rigidB2A ? xfNodeBoxes[n] : tree.nodes()[n].box; };
    auto leafBox = [&]( int i ) -> const Box3f & { return rigidB2A ? xfLeafBoxes[i] : tree.leafBoxes()[i]; };
    auto nodesIntersect = [&]( NodeId x, NodeId y ) { return preciseBoxesIntersect( nodeBox( x ), nodeBox( y ), conv, minGap ); };
    auto split = [&]( const WideNodePair & s, auto && push ) { return splitSelfWidePair( tree, s, nodesIntersect, push ); };
    // 8 levels -> at most 4^8 = 2^16 subtasks
    const auto subtasks = makeWideSubtasks( { { NodeId{ 0 }, NodeId{ 0 } } }, 8, split );

    std::vector<std::vector<EdgeTri>> subtaskRes( subtasks.size() );

    // we do not check an edge if its right triangle has smaller index and also in the mesh part
    auto checkEdge = [&] ( EdgeId e )
    {
        const auto r = mp.mesh.topology.right( e );
        if ( !r )
            return true;
        if ( mp.region && !mp.region->test( r ) )
            return true;

        const auto l = mp.mesh.topology.left( e );
        assert( l );
        return l < r;
    };

    auto checkTwoTris = [&] ( const AABBTree::LeafTri & lt, const AABBTree::LeafTri & rt, std::vector<EdgeTri>& res )
    {
        const FaceId lTri = lt.face;
        const FaceId rTri = rt.face;
        PreciseVertCoords lvc[3], rvc[3];
        for ( int j = 0; j < 3; ++j )
        {
            lvc[j].id = lt.verts[j];
            rvc[j].id = rt.verts[j];
        }

        // the triangles with two shared vertices (connected by a shared edge or by multiple edges) can meet only along the segment between them,
        // so they are skipped; otherwise find the indices of the shared vertex in both triangles
        std::pair<int, int> sharedVerts{ -1, -1 };
        int numSharedVerts = 0;
        for ( int i = 0; i < 3; ++i )
        {
            for ( int j = 0; j < 3; ++j )
            {
                if ( lvc[i].id == rvc[j].id )
                {
                    sharedVerts = { i, j };
                    ++numSharedVerts;
                }
            }
        }
        if ( numSharedVerts >= 2 )
            return;

        for ( int j = 0; j < 3; ++j )
        {
            const auto& lf = mp.mesh.points[lvc[j].id];
            lvc[j].pt = conv( rigidB2A ? ( *rigidB2A )( lf ) : lf );
            lvc[j].id += aVertsSize;

            const auto rf = mp.mesh.points[rvc[j].id];
            rvc[j].pt = conv( rigidB2A ? ( *rigidB2A )( rf ) : rf );
            rvc[j].id += aVertsSize;
        }

        // check edges from A
        EdgeId lEdge = mp.mesh.topology.edgeWithLeft( lTri );
        auto lEdgeCheck = [&] ( int v0, int v1 )
        {
            // skip incident to shared vert
            if ( sharedVerts.first == v0 || sharedVerts.first == v1 )
                return EdgeId{};
            if ( !checkEdge( lEdge ) )
                return EdgeId{};
            auto isect = doTriangleSegmentIntersect( { rvc[0], rvc[1], rvc[2], lvc[v0], lvc[v1] } );
            if ( !isect )
                return EdgeId{};
            return isect.dIsLeftFromABC ? lEdge : lEdge.sym();
        };
        if ( auto e = lEdgeCheck( 0, 1 ) )
        {
            res.emplace_back( e, rTri );
        }
        lEdge = mp.mesh.topology.prev( lEdge.sym() );
        if ( auto e = lEdgeCheck( 1, 2 ) )
        {
            res.emplace_back( e, rTri );
        }
        lEdge = mp.mesh.topology.prev( lEdge.sym() );
        if ( auto e = lEdgeCheck( 2, 0 ) )
        {
            res.emplace_back( e, rTri );
        }

        // check edges from B
        EdgeId rEdge = mp.mesh.topology.edgeWithLeft( rTri );
        auto rEdgeCheck = [&] ( int v0, int v1 )
        {
            // skip incident to shared vert
            if ( sharedVerts.second == v0 || sharedVerts.second == v1 )
                return EdgeId{};
            if ( !checkEdge( rEdge ) )
                return EdgeId{};
            auto isect = doTriangleSegmentIntersect( { lvc[0], lvc[1], lvc[2], rvc[v0], rvc[v1] } );
            if ( !isect )
                return EdgeId{};
            return isect.dIsLeftFromABC ? rEdge : rEdge.sym();
        };
        if ( auto e = rEdgeCheck( 0, 1 ) )
        {
            res.emplace_back( e, lTri );
        }
        rEdge = mp.mesh.topology.prev( rEdge.sym() );
        if ( auto e = rEdgeCheck( 1, 2 ) )
        {
            res.emplace_back( e, lTri );
        }
        rEdge = mp.mesh.topology.prev( rEdge.sym() );
        if ( auto e = rEdgeCheck( 2, 0 ) )
        {
            res.emplace_back( e, lTri );
        }
    };

    std::atomic<bool> keepGoing{ true };
    // checks subtasks in parallel
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, subtasks.size() ),
        [&] ( const tbb::blocked_range<size_t>& range )
    {
        Vector<WideNodePair> mySubtasks;
        for ( auto is = range.begin(); is < range.end(); ++is )
        {
            if ( anyIntersection && !keepGoing.load( std::memory_order_relaxed ) )
                break;
            mySubtasks.push_back( subtasks[is] );
            std::vector<EdgeTri> myRes;
            while ( !mySubtasks.empty() )
            {
                const auto s = mySubtasks.back();
                mySubtasks.pop_back();
                if ( split( s, [&mySubtasks]( const WideNodePair & c ) { mySubtasks.push_back( c ); } ) )
                    continue;
                const bool goOn = forWideLeafTris<true>( tree, tree, s,
                    [&]( int i, NodeId n ) { return preciseBoxesIntersect( leafBox( i ), nodeBox( n ), conv, minGap ); },
                    [&]( int i, int j ) { return preciseBoxesIntersect( leafBox( i ), leafBox( j ), conv, minGap ); },
                    [&]( int i, int j )
                    {
                        const auto & lt = tree.leafTris()[i];
                        if ( mp.region && !mp.region->test( lt.face ) )
                            return true;
                        const auto & rt = tree.leafTris()[j];
                        if ( mp.region && !mp.region->test( rt.face ) )
                            return true;
                        checkTwoTris( lt, rt, myRes );
                        return !anyIntersection || myRes.empty();
                    } );
                if ( !goOn )
                {
                    keepGoing.store( false, std::memory_order_relaxed );
                    mySubtasks.clear();
                    break;
                }
            }
            subtaskRes[is] = std::move( myRes );
        }
    } );

    // unite results from sub-trees into final vectors
    size_t cols = 0;
    for ( const auto& s : subtaskRes )
    {
        cols += s.size();
    }
    res.reserve( cols );
    for ( const auto& s : subtaskRes )
    {
        res.insert( res.end(), s.begin(), s.end() );
    }

    return res;

}

CoordinateConverters getVectorConverters( const MeshPart& a, const MeshPart& b, const AffineXf3f* rigidB2A )
{
    MR_TIMER;
    Box3d bb( a.mesh.computeBoundingBox( a.region ) );
    bb.include( Box3d( b.mesh.computeBoundingBox( b.region, rigidB2A ) ) );
    CoordinateConverters res;
    res.toInt = getToIntConverter( bb );
    res.toFloat = getToFloatConverter( bb );
    return res;
}

CoordinateConverters getVectorConverters( const MeshPart& a )
{
    MR_TIMER;
    Box3d bb( a.mesh.computeBoundingBox( a.region ) );
    CoordinateConverters res;
    res.toInt = getToIntConverter( bb );
    res.toFloat = getToFloatConverter( bb );
    return res;
}

} //namespace MR
