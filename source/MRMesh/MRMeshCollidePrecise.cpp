#include "MRMeshCollidePrecise.h"
#include "MRAABBTree.h"
#include "MRMesh.h"
#include "MRPrecisePredicates3.h"
#include "MRFaceFace.h"
#include "MRTimer.h"
#include "MRParallelFor.h"
#include "MRProcessSelfTreeSubtasks.h"
#include "MRRingIterator.h"
#include "MRBuffer.h"
#include <array>
#include <atomic>

namespace MR
{

namespace
{

/// tests the boxes of a pair of nodes; the pair is pushed in the stack unless both are leaves, then it is given to processLeaves;
/// returns true to stop the search
template <typename BoxesIntersect, typename ProcessLeaves>
MR_FORCE_INLINE bool processNodePair( const NodeNode & c, const AABBTree & aTree, const AABBTree & bTree,
    const BoxesIntersect & boxesIntersect, Vector<NodeNode> & stack, ProcessLeaves && processLeaves )
{
    if ( !boxesIntersect( c.aNode, c.bNode ) )
        return false;
    if ( !aTree.isLeaf( c.aNode ) || !bTree.isLeaf( c.bNode ) )
    {
        stack.push_back( c );
        return false;
    }
    return processLeaves( c );
}

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

/// the boxes of the nodes and leaves of a tree after the transformation xf (if given);
/// each transformed box is computed on the first request (from any thread) and kept for the next ones
class CachedBoxes
{
public:
    CachedBoxes( const AABBTree & tree, const AffineXf3f * xf )
        : tree_( tree ), xf_( xf ), firstLeaf_( tree.nodes().size() )
    {
        if ( !xf )
            return;
        const auto sz = firstLeaf_ + tree.numLeaves();
        boxes_.resize( sz );
        state_ = Vector<std::atomic<std::uint8_t>>( sz );
    }

    const Box3f & operator()( NodeId n ) const
    {
        if ( !xf_ )
            return tree_.box( n );
        const size_t i = AABBTree::isLeaf( n ) ? firstLeaf_ + AABBTree::Node::leafIndex( n ) : size_t( n );
        auto & state = state_[i];
        if ( state.load( std::memory_order_acquire ) == cReady )
            return boxes_[i];
        std::uint8_t expected = cEmpty;
        if ( state.compare_exchange_strong( expected, cWriting, std::memory_order_relaxed ) )
        {
            boxes_[i] = transformed( tree_.box( n ), *xf_ );
            state.store( cReady, std::memory_order_release );
        }
        else
        {
            // another thread is storing this box right now
            while ( state.load( std::memory_order_acquire ) != cReady )
                {}
        }
        return boxes_[i];
    }

private:
    static constexpr std::uint8_t cEmpty = 0, cWriting = 1, cReady = 2;
    const AABBTree & tree_;
    const AffineXf3f * xf_ = nullptr;
    size_t firstLeaf_ = 0;
    mutable Buffer<Box3f> boxes_; ///< nodes first, then leaves
    mutable Vector<std::atomic<std::uint8_t>> state_;
};

/// decides by preciseBoxesIntersect whether a node of tree A and a node of tree B after the transformation xf (if given) intersect
class PreciseBoxesIntersection
{
public:
    PreciseBoxesIntersection( const AABBTree & aTree, const AABBTree & bTree, const ConvertToIntVector & conv, const AffineXf3f * xf )
        : aTree_( aTree ), bBoxes_( bTree, xf ), conv_( conv ), minGap_( 2 / conv.invRange )
    {
    }

    bool operator()( NodeId a, NodeId b ) const
    {
        return preciseBoxesIntersect( aTree_.box( a ), bBoxes_( b ), conv_, minGap_ );
    }

private:
    const AABBTree & aTree_;
    CachedBoxes bBoxes_;
    const ConvertToIntVector & conv_;
    double minGap_ = 0;
};

/// finds all pairs of colliding edges and triangles of two mesh parts, given the AABB trees with (at least) all triangles of the parts
PreciseCollisionResult findCollidingEdgeTrisPrecise( const MeshPart & a, const AABBTree & aTree,
    const MeshPart & b, const AABBTree & bTree, const PreciseBoxesIntersection & boxesIntersect,
    ConvertToIntVector conv, const AffineXf3f * rigidB2A, bool anyIntersection )
{
    PreciseCollisionResult res;
    if ( aTree.empty() || bTree.empty() )
        return res;

    // sequentially subdivide full task on smaller subtasks;
    // they shall be not too many for this subdivision not to take too long;
    // and they shall be not too few for enough parallelism later
    Timer t( "2 top subtasks" );

    Vector<NodeNode> subtasks{ { aTree.rootNodeId(), bTree.rootNodeId() } }, nextSubtasks, leafTasks;
    // tested on two Spheres each with 3366 vertices (these numbers are outdated after preparation of precise boxes):
    // 16 -> init=0.886 (13%), main=5.948, total=6.834
    // 14 -> init=0.429 ( 7%), main=5.990, total=6.419
    // 12 -> init=0.226 ( 3%), main=6.445, total=6.671
    for( int i = 0; i < 14; ++i ) // 14 -> will produce at most 2^14 subtasks
    {
        int numSplits = 0;
        while( !subtasks.empty() )
        {
            const auto s = subtasks.back();
            subtasks.pop_back();

            if ( !boxesIntersect( s.aNode, s.bNode ) )
                continue;

            const bool aLeaf = aTree.isLeaf( s.aNode );
            const bool bLeaf = bTree.isLeaf( s.bNode );
            if ( aLeaf && bLeaf )
            {
                leafTasks.push_back( s );
                continue;
            }

            if ( !aLeaf && ( bLeaf || aTree.box( s.aNode ).volume() >= bTree.box( s.bNode ).volume() ) )
            {
                // split aNode
                const auto & aNode = aTree[s.aNode];
                nextSubtasks.push_back( { aNode.l, s.bNode } );
                nextSubtasks.push_back( { aNode.r, s.bNode } );
            }
            else
            {
                assert( !bLeaf );
                // split bNode
                const auto & bNode = bTree[s.bNode];
                nextSubtasks.push_back( { s.aNode, bNode.l } );
                nextSubtasks.push_back( { s.aNode, bNode.r } );
            }
            ++numSplits;
        }
        subtasks.swap( nextSubtasks );
        if ( !numSplits )
            break;
    }
    subtasks.insert( subtasks.end(), leafTasks.begin(), leafTasks.end() );

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

    const int aVertsSize = (int)a.mesh.topology.vertSize();
    auto checkTwoTris = [&]( NodeId aLeaf, NodeId bLeaf, PreciseCollisionResult & res )
    {
        const auto aTri = aTree.leafId( aLeaf );
        const auto bTri = bTree.leafId( bLeaf );
        PreciseVertCoords avc[3], bvc[3];
        for ( int j = 0; j < 3; ++j )
        {
            avc[j].id = aTree.leafVerts( aLeaf )[j];
            bvc[j].id = bTree.leafVerts( bLeaf )[j];
            avc[j].pt = conv( a.mesh.points[avc[j].id] );
            const auto bf = b.mesh.points[bvc[j].id];
            bvc[j].pt = conv( rigidB2A ? (*rigidB2A)( bf ) : bf );
            bvc[j].id += aVertsSize;
        }

        // check edges from A
        int numA = 0;
        EdgeId aEdge = a.mesh.topology.edgeWithLeft( aTri );
        auto aEdgeCheck = [&]( int v0, int v1 )
        {
            if ( !checkEdge( aEdge, a ) )
                return EdgeId{};
            auto isect = doTriangleSegmentIntersect( { bvc[0], bvc[1], bvc[2], avc[v0], avc[v1] } );
            if ( !isect )
                return EdgeId{};
            return isect.dIsLeftFromABC ? aEdge : aEdge.sym();
        };
        if ( auto e = aEdgeCheck( 0, 1 ) )
        {
            res.emplace_back( true, e, bTri );
        }
        aEdge = a.mesh.topology.prev( aEdge.sym() );
        if ( auto e = aEdgeCheck( 1, 2 ) )
        {
            res.emplace_back( true, e, bTri );
        }
        aEdge = a.mesh.topology.prev( aEdge.sym() );
        if ( numA < 2 )
        {
            if ( auto e = aEdgeCheck( 2, 0 ) )
                res.emplace_back( true, e, bTri );
        }

        // check edges from B
        int numB = 0;
        EdgeId bEdge = b.mesh.topology.edgeWithLeft( bTri );
        auto bEdgeCheck = [&]( int v0, int v1 )
        {
            if ( !checkEdge( bEdge, b ) )
                return EdgeId{};
            auto isect = doTriangleSegmentIntersect( { avc[0], avc[1], avc[2], bvc[v0], bvc[v1] } );
            if ( !isect )
                return EdgeId{};
            return isect.dIsLeftFromABC ? bEdge : bEdge.sym();
        };
        if ( auto e = bEdgeCheck( 0, 1 ) )
        {
            res.emplace_back( false, e, aTri );
        }
        bEdge = b.mesh.topology.prev( bEdge.sym() );
        if ( auto e = bEdgeCheck( 1, 2 ) )
        {
            res.emplace_back( false, e, aTri );
        }
        bEdge = b.mesh.topology.prev( bEdge.sym() );
        if ( numB < 2 )
        {
            if ( auto e = bEdgeCheck( 2, 0 ) )
                res.emplace_back( false, e, aTri );
        }
    };

    // checks subtasks in parallel
    t.restart( "3 process" );

    struct ThreadData
    {
        PreciseCollisionResult res;
        Vector<NodeNode> subtasks;
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
        Vector<NodeNode>& mySubtasks = tls.subtasks;
        assert( mySubtasks.empty() );
        SubtaskRes myRes{ .vec = &tls.res };
        myRes.first = (int)myRes.vec->size();
        // returns true to stop the search
        auto processLeaves = [&]( const NodeNode & s )
        {
            const auto aFace = aTree.leafId( s.aNode );
            if ( a.region && !a.region->test( aFace ) )
                return false;
            const auto bFace = bTree.leafId( s.bNode );
            if ( b.region && !b.region->test( bFace ) )
                return false;
            checkTwoTris( s.aNode, s.bNode, *myRes.vec );
            if ( anyIntersection && !myRes.vec->empty() )
            {
                anyIntersectionAtm.store( true, std::memory_order_relaxed );
                return true;
            }
            return false;
        };
        // the boxes of a pair are tested before it is pushed, and a pair of leaves is processed at once
        if ( !processNodePair( subtasks[is], aTree, bTree, boxesIntersect, mySubtasks, processLeaves ) )
        {
            while ( !mySubtasks.empty() )
            {
                if ( anyIntersection && anyIntersectionAtm.load( std::memory_order_relaxed ) )
                    break;
                const auto s = mySubtasks.back();
                mySubtasks.pop_back();

                const bool aLeaf = aTree.isLeaf( s.aNode );
                const bool bLeaf = bTree.isLeaf( s.bNode );
                assert( !aLeaf || !bLeaf );
                if ( !aLeaf && ( bLeaf || aTree.box( s.aNode ).volume() >= bTree.box( s.bNode ).volume() ) )
                {
                    // split aNode
                    const auto & aNode = aTree[s.aNode];
                    if ( processNodePair( { aNode.l, s.bNode }, aTree, bTree, boxesIntersect, mySubtasks, processLeaves )
                      || processNodePair( { aNode.r, s.bNode }, aTree, bTree, boxesIntersect, mySubtasks, processLeaves ) )
                        break;
                }
                else
                {
                    // split bNode
                    const auto & bNode = bTree[s.bNode];
                    if ( processNodePair( { s.aNode, bNode.l }, aTree, bTree, boxesIntersect, mySubtasks, processLeaves )
                      || processNodePair( { s.aNode, bNode.r }, aTree, bTree, boxesIntersect, mySubtasks, processLeaves ) )
                        break;
                }
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
    if ( tree.empty() )
        return res;

    // the traversal visits every node, and each many times, so the boxes after the transformation are computed once in advance
    AABBTree::Boxes xfBoxes; // in the order of tree.boxes()
    if ( rigidB2A )
    {
        xfBoxes.resizeNoInit( tree.boxes().size() );
        ParallelFor( xfBoxes, [&]( size_t i ) { xfBoxes[i] = transformed( tree.boxes()[i], *rigidB2A ); } );
    }
    const auto numLeaves = std::ptrdiff_t( tree.numLeaves() );
    auto xfBox = [&]( NodeId n ) -> const Box3f &
    {
        if ( !rigidB2A )
            return tree.box( n );
        return xfBoxes.data()[numLeaves + int( n )];
    };
    const double minGap = 2 / conv.invRange;
    auto processBoxes = [&] ( const NodeNode & s )
    {
        const Box3f & aBox = xfBox( s.aNode );
        const Box3f & bBox = xfBox( s.bNode );
        return preciseBoxesIntersect( aBox, bBox, conv, minGap ) ? Processing::Continue : Processing::Stop;
    };

    // sequentially subdivide full task on smaller subtasks;
    // they shall be not too many for this subdivision not to take too long;
    // and they shall be not too few for enough parallelism later
    std::vector<NodeNode> subtasks{ { tree.rootNodeId(), tree.rootNodeId() } }, nextSubtasks, leafTasks;
    for ( int i = 0; i < 16 && !subtasks.empty(); ++i ) // 16 -> will produce at most 2^16 subtasks
    {
        processSelfSubtasks( tree, subtasks, nextSubtasks,
            [&leafTasks]( const NodeNode & s ) { leafTasks.push_back( s ); return Processing::Continue; }, processBoxes );
        subtasks.swap( nextSubtasks );
    }
    subtasks.insert( subtasks.end(), leafTasks.begin(), leafTasks.end() );

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

    auto checkTwoTris = [&] ( NodeId lLeaf, NodeId rLeaf, std::vector<EdgeTri>& res )
    {
        const auto lTri = tree.leafId( lLeaf );
        const auto rTri = tree.leafId( rLeaf );
        PreciseVertCoords lvc[3], rvc[3];
        for ( int j = 0; j < 3; ++j )
        {
            lvc[j].id = tree.leafVerts( lLeaf )[j];
            rvc[j].id = tree.leafVerts( rLeaf )[j];
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
        std::vector<NodeNode> mySubtasks;
        for ( auto is = range.begin(); is < range.end(); ++is )
        {
            mySubtasks.push_back( subtasks[is] );
            std::vector<EdgeTri> myRes;
            processSelfSubtasks( tree, mySubtasks, mySubtasks, 
                [&tree, &mp, &myRes, &checkTwoTris, anyIntersection, &keepGoing] ( const NodeNode& s )
            {
                const auto lFace = tree.leafId( s.aNode );
                if ( mp.region && !mp.region->test( lFace ) )
                    return Processing::Continue;
                const auto rFace = tree.leafId( s.bNode );
                if ( mp.region && !mp.region->test( rFace ) )
                    return Processing::Continue;
                checkTwoTris( s.aNode, s.bNode, myRes );
                if ( anyIntersection && !myRes.empty() )
                {
                    keepGoing.store( false, std::memory_order_relaxed );
                    return Processing::Stop;
                }
                return Processing::Continue;
            }, 
            processBoxes );
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
