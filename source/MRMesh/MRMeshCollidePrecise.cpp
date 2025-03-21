#include "MRMeshCollidePrecise.h"
#include "MRAABBTree.h"
#include "MRMesh.h"
#include "MRPrecisePredicates3.h"
#include "MRFaceFace.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"
#include "MRProcessSelfTreeSubtasks.h"
#include <array>

namespace MR
{

PreciseCollisionResult findCollidingEdgeTrisPrecise( const MeshPart & a, const MeshPart & b, 
    ConvertToIntVector conv, const AffineXf3f * rigidB2A, bool anyIntersection )
{
    MR_TIMER;

    PreciseCollisionResult res;
    const AABBTree & aTree = a.mesh.getAABBTree();
    const AABBTree & bTree = b.mesh.getAABBTree();
    if ( aTree.nodes().empty() || bTree.nodes().empty() )
        return res;

    // sequentially subdivide full task on smaller subtasks;
    // they shall be not too many for this subdivision not to take too long;
    // and they shall be not too few for enough parallelism later
    std::vector<NodeNode> subtasks{ { NodeId{ 0 }, NodeId{ 0 } } }, nextSubtasks, leafTasks;
    for( int i = 0; i < 16; ++i ) // 16 -> will produce at most 2^16 subtasks
    {
        int numSplits = 0;
        while( !subtasks.empty() )
        {
            const auto s = subtasks.back();
            subtasks.pop_back();
            const auto & aNode = aTree[s.aNode];
            const auto & bNode = bTree[s.bNode];

            // check intersection in int boxes for consistency with precise intersections
            auto transformedBoxb = transformed( bNode.box, rigidB2A );
            Box3i aBox{ conv( aNode.box.min ),conv( aNode.box.max ) };
            Box3i bBox{ conv( transformedBoxb.min ),conv( transformedBoxb.max ) };
            if ( !aBox.intersects( bBox ) )
                continue;

            if ( aNode.leaf() && bNode.leaf() )
            {
                leafTasks.push_back( s );
                continue;
            }
        
            if ( !aNode.leaf() && ( bNode.leaf() || aNode.box.volume() >= bNode.box.volume() ) )
            {
                // split aNode
                nextSubtasks.emplace_back( aNode.l, s.bNode );
                nextSubtasks.emplace_back( aNode.r, s.bNode );
            }
            else
            {
                assert( !bNode.leaf() );
                // split bNode
                nextSubtasks.emplace_back( s.aNode, bNode.l );
                nextSubtasks.emplace_back( s.aNode, bNode.r );
            }
            ++numSplits;
        }
        subtasks.swap( nextSubtasks );
        if ( !numSplits )
            break;
    }
    subtasks.insert( subtasks.end(), leafTasks.begin(), leafTasks.end() );

    std::vector<PreciseCollisionResult> subtaskRes( subtasks.size() );

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
    auto checkTwoTris = [&]( FaceId aTri, FaceId bTri, PreciseCollisionResult & res )
    {
        PreciseVertCoords avc[3], bvc[3];
        a.mesh.topology.getTriVerts( aTri, avc[0].id, avc[1].id, avc[2].id );
        b.mesh.topology.getTriVerts( bTri, bvc[0].id, bvc[1].id, bvc[2].id );

        for ( int j = 0; j < 3; ++j )
        {
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
            res.edgesAtrisB.emplace_back( e, bTri );
        }
        aEdge = a.mesh.topology.prev( aEdge.sym() );
        if ( auto e = aEdgeCheck( 1, 2 ) )
        {
            res.edgesAtrisB.emplace_back( e, bTri );
        }
        aEdge = a.mesh.topology.prev( aEdge.sym() );
        if ( numA < 2 )
        {
            if ( auto e = aEdgeCheck( 2, 0 ) )
                res.edgesAtrisB.emplace_back( e, bTri );
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
            res.edgesBtrisA.emplace_back( e, aTri );
        }
        bEdge = b.mesh.topology.prev( bEdge.sym() );
        if ( auto e = bEdgeCheck( 1, 2 ) )
        {
            res.edgesBtrisA.emplace_back( e, aTri );
        }
        bEdge = b.mesh.topology.prev( bEdge.sym() );
        if ( numB < 2 )
        {
            if ( auto e = bEdgeCheck( 2, 0 ) )
                res.edgesBtrisA.emplace_back( e, aTri );
        }
    };

    std::atomic<bool> anyIntersectionAtm{ false };
    // checks subtasks in parallel
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, subtasks.size() ),
        [&]( const tbb::blocked_range<size_t>& range )
    {
        std::vector<NodeNode> mySubtasks;
        for ( auto is = range.begin(); is < range.end(); ++is )
        {
            mySubtasks.push_back( subtasks[is] );
            PreciseCollisionResult myRes;
            while ( !mySubtasks.empty() )
            {
                if ( anyIntersection && anyIntersectionAtm.load( std::memory_order_relaxed ) )
                    break;
                const auto s = mySubtasks.back();
                mySubtasks.pop_back();
                const auto & aNode = aTree[s.aNode];
                const auto & bNode = bTree[s.bNode];

                // check intersection in int boxes for consistency with precise intersections
                auto transformedBoxb = transformed( bNode.box, rigidB2A );
                Box3i aBox{ conv( aNode.box.min ),conv( aNode.box.max ) };
                Box3i bBox{ conv( transformedBoxb.min ),conv( transformedBoxb.max ) };
                if ( !aBox.intersects( bBox ) )
                    continue;

                if ( aNode.leaf() && bNode.leaf() )
                {
                    const auto aFace = aNode.leafId();
                    if ( a.region && !a.region->test( aFace ) )
                        continue;
                    const auto bFace = bNode.leafId();
                    if ( b.region && !b.region->test( bFace ) )
                        continue;
                    checkTwoTris( aFace, bFace, myRes );
                    if ( anyIntersection && ( !myRes.edgesAtrisB.empty() || !myRes.edgesBtrisA.empty() ) )
                    {
                        anyIntersectionAtm.store( true, std::memory_order_relaxed );
                        break;
                    }
                    continue;
                }
        
                if ( !aNode.leaf() && ( bNode.leaf() || aNode.box.volume() >= bNode.box.volume() ) )
                {
                    // split aNode
                    mySubtasks.emplace_back( aNode.l, s.bNode );
                    mySubtasks.emplace_back( aNode.r, s.bNode );
                }
                else
                {
                    assert( !bNode.leaf() );
                    // split bNode
                    mySubtasks.emplace_back( s.aNode, bNode.l );
                    mySubtasks.emplace_back( s.aNode, bNode.r );
                }
            }
            subtaskRes[is] = std::move( myRes );
        }
    } );

    // unite results from sub-trees into final vectors
    size_t colsAB = 0, colsBA = 0;
    for ( const auto & s : subtaskRes )
    {
        colsAB += s.edgesAtrisB.size();
        colsBA += s.edgesBtrisA.size();
    }
    res.edgesAtrisB.reserve( colsAB );
    res.edgesBtrisA.reserve( colsBA );
    for ( const auto & s : subtaskRes )
    {
        res.edgesAtrisB.insert( res.edgesAtrisB.end(), s.edgesAtrisB.begin(), s.edgesAtrisB.end() );
        res.edgesBtrisA.insert( res.edgesBtrisA.end(), s.edgesBtrisA.begin(), s.edgesBtrisA.end() );
    }

    return res;
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


inline std::pair<int, int> sharedPreciseVertCoord( const PreciseVertCoords av[3], const PreciseVertCoords bv[3] )
{
    for ( int i = 0; i < 3; ++i )
    {
        for ( int j = 0; j < 3; ++j )
        {
            if ( av[i].id == bv[j].id )
                return { i, j };
        }
    }
    return { -1, -1 };
}

std::vector<EdgeTri> findSelfCollidingEdgeTrisPrecise( const MeshPart& mp, ConvertToIntVector conv, bool anyIntersection /*= false */, 
    const AffineXf3f* rigidB2A, int aVertsSize )
{
    MR_TIMER;

    std::vector<EdgeTri> res;
    const AABBTree& tree = mp.mesh.getAABBTree();
    if ( tree.nodes().empty() )
        return res;

    auto processBoxes = [&rigidB2A,&conv] ( const Box3f& lBox, const Box3f& rBox )
    {
        auto transformedLBox = transformed( lBox, rigidB2A );
        auto transformedRBox = transformed( rBox, rigidB2A );
        Box3i liBox{ conv( transformedLBox.min ),conv( transformedLBox.max ) };
        Box3i riBox{ conv( transformedRBox.min ),conv( transformedRBox.max ) };
        return liBox.intersects( riBox ) ? Processing::Continue : Processing::Stop;
    };

    // sequentially subdivide full task on smaller subtasks;
    // they shall be not too many for this subdivision not to take too long;
    // and they shall be not too few for enough parallelism later
    std::vector<NodeNode> subtasks{ { NodeId{ 0 }, NodeId{ 0 } } }, nextSubtasks, leafTasks;
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

    auto checkTwoTris = [&] ( FaceId lTri, FaceId rTri, std::vector<EdgeTri>& res )
    {
        PreciseVertCoords lvc[3], rvc[3];
        mp.mesh.topology.getTriVerts( lTri, lvc[0].id, lvc[1].id, lvc[2].id );
        mp.mesh.topology.getTriVerts( rTri, rvc[0].id, rvc[1].id, rvc[2].id );

        for ( int j = 0; j < 3; ++j )
        {
            const auto& lf = mp.mesh.points[lvc[j].id];
            lvc[j].pt = conv( rigidB2A ? ( *rigidB2A )( lf ) : lf );
            lvc[j].id += aVertsSize;

            const auto rf = mp.mesh.points[rvc[j].id];
            rvc[j].pt = conv( rigidB2A ? ( *rigidB2A )( rf ) : rf );
            rvc[j].id += aVertsSize;
        }

        auto sharedVerts = sharedPreciseVertCoord( lvc, rvc );

        // check edges from A
        int numL = 0;
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
        if ( numL < 2 )
        {
            if ( auto e = lEdgeCheck( 2, 0 ) )
                res.emplace_back( e, rTri );
        }

        // check edges from B
        int numR = 0;
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
            return isect.dIsLeftFromABC ? rEdge : rEdge.sym(); // TODO: need flip?
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
        if ( numR < 2 )
        {
            if ( auto e = rEdgeCheck( 2, 0 ) )
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
                const auto& lNode = tree[s.aNode];
                const auto& rNode = tree[s.bNode];
                const auto lFace = lNode.leafId();
                if ( mp.region && !mp.region->test( lFace ) )
                    return Processing::Continue;
                const auto rFace = rNode.leafId();
                if ( mp.region && !mp.region->test( rFace ) )
                    return Processing::Continue;
                if ( mp.mesh.topology.sharedEdge( lFace, rFace ) )
                    return Processing::Continue;
                checkTwoTris( lFace, rFace, myRes );
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
    Box3d bb;
    bb.include( Box3d( a.mesh.computeBoundingBox() ) );
    Box3f bMeshBox = b.mesh.computeBoundingBox( rigidB2A );
    bb.include( Box3d( bMeshBox ) );
    CoordinateConverters res;
    res.toInt = getToIntConverter( bb );
    res.toFloat = getToFloatConverter( bb );
    return res;
}

} //namespace MR
