#include "MRMeshCollide.h"
#include "MRAABBTree.h"
#include "MRMesh.h"
#include "MRParallelFor.h"
#include "MRTriangleIntersection.h"
#include "MREnums.h"
#include "MRTimer.h"
#include "MRTriMath.h"
#include "MRTriDist.h"
#include "MRExpected.h"
#include "MRProcessSelfTreeSubtasks.h"
#include "MRMeshProject.h"
#include "MRPrecisePredicates3.h"
#include "MRRingIterator.h"

#include <atomic>
#include <thread>

namespace MR
{

std::vector<FaceFace> findCollidingTriangles( const MeshPart & a, const MeshPart & b, const AffineXf3f * rigidB2A, bool firstIntersectionOnly )
{
    MR_TIMER;

    std::vector<FaceFace> res;
    const AABBTree & aTree = a.mesh.getAABBTree();
    const AABBTree & bTree = b.mesh.getAABBTree();
    if ( aTree.nodes().empty() || bTree.nodes().empty() )
        return res;

    NodeBitSet aNodes, bNodes;
    NodeBitSet* aNodesPtr{nullptr}, * bNodesPtr{nullptr};
    if ( a.region )
    {
        aNodes = aTree.getNodesFromLeaves( *a.region );
        aNodesPtr = &aNodes;
    }
    if ( b.region )
    {
        bNodes = bTree.getNodesFromLeaves( *b.region );
        bNodesPtr = &bNodes;
    }

    std::vector<NodeNode> subtasks{ { NodeId{ 0 }, NodeId{ 0 } } };

    while( !subtasks.empty() )
    {
        const auto s = subtasks.back();
        subtasks.pop_back();

        if ( aNodesPtr && !aNodes.test( s.aNode ) )
            continue;
        if ( bNodesPtr && !bNodes.test( s.bNode ) )
            continue;

        const auto & aNode = aTree[s.aNode];
        const auto & bNode = bTree[s.bNode];

        const auto overlap = aNode.box.intersection( transformed( bNode.box, rigidB2A ) );
        if ( !overlap.valid() )
            continue;

        if ( aNode.leaf() && bNode.leaf() )
        {
            const auto aFace = aNode.leafId();
            const auto bFace = bNode.leafId();
            res.emplace_back( aFace, bFace );
            continue;
        }

        if ( !aNode.leaf() && ( bNode.leaf() || aNode.box.volume() >= bNode.box.volume() ) )
        {
            // split aNode
            subtasks.push_back( { aNode.l, s.bNode } );
            subtasks.push_back( { aNode.r, s.bNode } );
        }
        else
        {
            assert( !bNode.leaf() );
            // split bNode
            subtasks.push_back( { s.aNode, bNode.l } );
            subtasks.push_back( { s.aNode, bNode.r } );
        }
    }

    std::atomic<int> firstIntersection{ (int)res.size() };
    ParallelFor( res, [&] ( size_t i )
    {
        int knownIntersection = firstIntersection.load( std::memory_order_relaxed );
        if ( firstIntersectionOnly && knownIntersection < i )
            return;
        Vector3f av[3], bv[3];
        a.mesh.getTriPoints( res[i].aFace, av[0], av[1], av[2] );
        b.mesh.getTriPoints( res[i].bFace, bv[0], bv[1], bv[2] );
        if ( rigidB2A )
        {
            bv[0] = (*rigidB2A)( bv[0] );
            bv[1] = (*rigidB2A)( bv[1] );
            bv[2] = (*rigidB2A)( bv[2] );
        }
        if ( doTrianglesIntersect( Vector3d{ av[0] }, Vector3d{ av[1] }, Vector3d{ av[2] }, Vector3d{ bv[0] }, Vector3d{ bv[1] }, Vector3d{ bv[2] } ) )
        {
            if ( firstIntersectionOnly )
            {
                while ( knownIntersection > i && !firstIntersection.compare_exchange_strong( knownIntersection, (int)i ) ) { }
                return;
            }
        }
        else
        {
            res[i].aFace = FaceId{}; //invalidate
        }
    } );

    if ( firstIntersectionOnly )
    {
        int knownIntersection = firstIntersection.load( std::memory_order_relaxed );
        if ( knownIntersection < res.size() )
        {
            res[0] = res[knownIntersection];
            res.erase( res.begin() + 1, res.end() );
        }
        else
            res.clear();
    }
    else
    {
        res.erase( std::remove_if( res.begin(), res.end(), []( const FaceFace & ff ) { return !ff.aFace.valid(); } ), res.end() );
    }

    return res;
}

std::pair<FaceBitSet, FaceBitSet> findCollidingTriangleBitsets( const MeshPart& a, const MeshPart& b,
    const AffineXf3f* rigidB2A )
{
    const auto pairs = findCollidingTriangles( a, b, rigidB2A );
    FaceId aMax, bMax;
    for ( const auto & p : pairs )
    {
        aMax = std::max( aMax, p.aFace );
        bMax = std::max( bMax, p.bFace );
    }

    std::pair<FaceBitSet, FaceBitSet> res;
    res.first.resize( aMax + 1 );
    res.second.resize( bMax + 1 );
    for ( const auto & p : pairs )
    {
        res.first.set( p.aFace );
        res.second.set( p.bFace );
    }
    return res;
}

inline std::pair<int, int> sharedVertex( const VertId av[3], const VertId bv[3] )
{
    for ( int i = 0; i < 3; ++i )
    {
        for ( int j = 0; j < 3; ++j )
        {
            if ( av[i] == bv[j] )
                return { i, j };
        }
    }
    return { -1, -1 };
}

Expected<bool> findSelfCollidingTriangles(
    const MeshPart& mp,
    std::vector<FaceFace> * outCollidingPairs,
    ProgressCallback cb,
    const Face2RegionMap * regionMap,
    bool touchIsIntersection )
{
    MR_TIMER;
    const AABBTree & tree = mp.mesh.getAABBTree();
    if ( tree.nodes().empty() )
        return false;

    auto sb = subprogress( cb, 0, 0.08f );

    // sequentially subdivide full task on smaller subtasks;
    // they shall be not too many for this subdivision not to take too long;
    // and they shall be not too few for enough parallelism later
    std::vector<NodeNode> subtasks{ { NodeId{ 0 }, NodeId{ 0 } } }, nextSubtasks, leafTasks;
    for( int i = 0; i < 16 && !subtasks.empty(); ++i ) // 16 -> will produce at most 2^16 subtasks
    {
        processSelfSubtasks( tree, subtasks, nextSubtasks,
            [&leafTasks]( const NodeNode & s ) { leafTasks.push_back( s ); return Processing::Continue; },
            [](const Box3f& lBox, const Box3f& rBox ){ return lBox.intersects( rBox ) ? Processing::Continue : Processing::Stop; });
        subtasks.swap( nextSubtasks );

        if ( !reportProgress( sb, i / 16.0f ) )
            return unexpectedOperationCanceled();
    }
    subtasks.insert( subtasks.end(), leafTasks.begin(), leafTasks.end() );

    sb = subprogress( cb, 0.08f, 0.92f );

    std::vector<std::vector<FaceFace>> subtaskRes( subtasks.size() );

    auto mainThreadId = std::this_thread::get_id();
    std::atomic<bool> keepGoing{ true };
    std::atomic<size_t> numDone;
    // checks subtasks in parallel
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, subtasks.size() ),
        [&]( const tbb::blocked_range<size_t>& range )
    {
        std::vector<NodeNode> mySubtasks;
        for ( auto is = range.begin(); is < range.end(); ++is )
        {
            if ( sb && !keepGoing.load( std::memory_order_relaxed ) )
                break;

            mySubtasks.push_back( subtasks[is] );
            std::vector<FaceFace> myRes;
            processSelfSubtasks( tree, mySubtasks, mySubtasks,
                [&tree, &mp, &myRes, regionMap, outCollidingPairs, &keepGoing, touchIsIntersection]( const NodeNode & s )
                {
                    const auto & aNode = tree[s.aNode];
                    const auto & bNode = tree[s.bNode];
                    const auto aFace = aNode.leafId();
                    if ( mp.region && !mp.region->test( aFace ) )
                        return Processing::Continue;
                    const auto bFace = bNode.leafId();
                    if ( mp.region && !mp.region->test( bFace ) )
                        return Processing::Continue;
                    if ( regionMap && ( *regionMap )[aFace] != ( *regionMap )[bFace] )
                        return Processing::Continue;

                    VertId av[3], bv[3];
                    Triangle3d ap, bp;

                    auto se = mp.mesh.topology.sharedEdge( aFace, bFace );
                    if ( se )
                    {
                        mp.mesh.topology.getLeftTriVerts( se, av[0], av[1], av[2] );
                        mp.mesh.topology.getLeftTriVerts( se.sym(), bv[0], bv[1], bv[2] );
                    }
                    else
                    {
                        mp.mesh.topology.getTriVerts( aFace, av[0], av[1], av[2] );
                        mp.mesh.topology.getTriVerts( bFace, bv[0], bv[1], bv[2] );
                    }
                    for ( int j = 0; j < 3; ++j )
                    {
                        ap[j] = Vector3d{ mp.mesh.points[av[j]] };
                        bp[j] = Vector3d{ mp.mesh.points[bv[j]] };
                    }
                    if ( se )
                    {
                        if ( !touchIsIntersection )
                            return Processing::Continue; // triangles sharing an edge may only touch one another

                        const auto na = normal( ap );
                        const auto nb = normal( bp );
                        constexpr auto epsSq = sqr( 1e-5 ); // angle must less than 5.73e-4 deg to consider two faces coplanar
                        if ( cross( na, nb ).lengthSq() > epsSq )
                            return Processing::Continue; // triangles are not coplanar

                        if ( dot( na, nb ) > 0 )
                            return Processing::Continue; // triangles are coplanar, but same-oriented, so they are separated by the shared edge

                        // triangles overlap in one plane
                    }
                    else if ( auto sv = sharedVertex( av, bv ); sv.first >= 0 )
                    {
                        // shared vertex
                        const int j = sv.first;
                        const int k = sv.second;
                        if ( !doTriangleSegmentIntersect( ap[0], ap[1], ap[2], bp[( k + 1 ) % 3], bp[( k + 2 ) % 3] ) &&
                             !doTriangleSegmentIntersect( bp[0], bp[1], bp[2], ap[( j + 1 ) % 3], ap[( j + 2 ) % 3] ) )
                        {
                            // check touching too
                            if ( !touchIsIntersection ||
                                  ( !isPointInTriangle( ap[( j + 1 ) % 3], bp[0], bp[1], bp[2] ) &&
                                    !isPointInTriangle( ap[( j + 2 ) % 3], bp[0], bp[1], bp[2] ) &&
                                    !isPointInTriangle( bp[( k + 1 ) % 3], ap[0], ap[1], ap[2] ) &&
                                    !isPointInTriangle( bp[( k + 2 ) % 3], ap[0], ap[1], ap[2] ) ) )
                                return Processing::Continue;
                            // else not touching
                        }
                    }
                    else if ( auto td = findTriTriDistance( { ap[0], ap[1], ap[2] }, { bp[0], bp[1], bp[2] }, { .upDistLimitSq = 0, .upLimitCheck = touchIsIntersection ? UpLimitCheck::Greater : UpLimitCheck::GreaterOrEqual } );
                        td.distSq > 0 || ( !touchIsIntersection && !td.overlap ) )
                    {
                        return Processing::Continue;
                    }
                    myRes.emplace_back( aFace, bFace );
                    if ( !outCollidingPairs )
                    {
                        keepGoing.store( false, std::memory_order_relaxed );
                        return Processing::Stop;
                    }
                    return Processing::Continue;
                },
                [](const Box3f& lBox, const Box3f& rBox ){ return lBox.intersects( rBox ) ? Processing::Continue : Processing::Stop; }
            );

            subtaskRes[is] = std::move( myRes );
        }

        if ( cb )
            numDone += range.size();

        if ( sb && std::this_thread::get_id() == mainThreadId )
        {
            if ( !reportProgress( sb, float( numDone ) / subtasks.size() ) )
                keepGoing.store( false, std::memory_order_relaxed );
        }
    } );

    // unite results from sub-trees into final vector
    size_t cols = 0;
    for ( const auto & s : subtaskRes )
        cols += s.size();

    if ( !outCollidingPairs && cols > 0 )
        return true; // even if keepGoing = false

    if ( !keepGoing.load( std::memory_order_relaxed ) || !reportProgress( sb, 1.0f ) )
        return unexpectedOperationCanceled();

    if ( outCollidingPairs )
    {
        outCollidingPairs->reserve( outCollidingPairs->size() + cols );
        for ( const auto & s : subtaskRes )
            outCollidingPairs->insert( outCollidingPairs->end(), s.begin(), s.end() );
    }

    if ( !reportProgress( cb, 1.0f ) )
        return unexpectedOperationCanceled();

    return cols > 0;
}

Expected<std::vector<FaceFace>> findSelfCollidingTriangles( const MeshPart& mp, ProgressCallback cb,
    const Face2RegionMap* regionMap,
    bool touchIsIntersection )
{
    std::vector<FaceFace> res;
    auto exp = findSelfCollidingTriangles( mp, &res, cb, regionMap, touchIsIntersection );
    if ( !exp )
        return unexpected( std::move( exp.error() ) );
    return res;
}

Expected<FaceBitSet> findSelfCollidingTrianglesBS( const MeshPart& mp, ProgressCallback cb, const Face2RegionMap* regionMap, bool touchIsIntersection )
{
    MR_TIMER;

    auto ffs = findSelfCollidingTriangles( mp, cb, regionMap, touchIsIntersection );
    if ( !ffs.has_value() )
        return unexpected( ffs.error() );

    FaceBitSet res;
    for ( const auto & ff : ffs.value() )
    {
        res.autoResizeSet( ff.aFace );
        res.autoResizeSet( ff.bFace );
    }

    return res;
}

bool isInside( const MeshPart & a, const MeshPart & b, const AffineXf3f * rigidB2A )
{
    auto cols = findCollidingTriangles( a, b, rigidB2A, true );
    if ( !cols.empty() )
        return false; // meshes intersect

    return isNonIntersectingInside( a, b, rigidB2A );
}

bool isNonIntersectingInside( const MeshPart& a, const MeshPart& b, const AffineXf3f* rigidB2A )
{
    auto aFace = a.mesh.topology.getFaceIds( a.region ).find_first();
    return isNonIntersectingInside( a.mesh, aFace, b, rigidB2A );
}

bool isNonIntersectingInside( const Mesh& a, FaceId aFace, const MeshPart& b, const AffineXf3f* rigidB2A /*= nullptr */ )
{
    if ( !aFace )
        return true; //consider empty mesh always inside

    Vector3f aPoint = a.triCenter( aFace );
    if ( rigidB2A )
        aPoint = rigidB2A->inverse()( aPoint );

    auto signDist = b.mesh.signedDistance( aPoint, FLT_MAX, b.region );
    return signDist && signDist < 0;
}

namespace
{

// true if given point is on the side of the plane of given face that its outer normal points away from
bool isBehindFacePrecise( const Mesh& m, FaceId f, const PreciseVertCoords& p,
    const CoordinateConverters& conv, int vertShift, const AffineXf3f* xf )
{
    std::array<PreciseVertCoords, 4> vs;
    m.topology.getTriVerts( f, vs[0].id, vs[1].id, vs[2].id );
    for ( int i = 0; i < 3; ++i )
    {
        const auto& q = m.points[vs[i].id];
        vs[i].pt = conv.toInt( xf ? ( *xf )( q ) : q );
        vs[i].id = VertId( int( vs[i].id ) + vertShift );
    }
    vs[3] = p;
    return orient3d( vs );
}

// true if the material of the mesh near given edge is not wider than a half-space;
// both faces of the edge must be present;
// all four points here are from the same mesh, so no shift of their ids is necessary
bool isConvexEdgePrecise( const Mesh& m, EdgeId e, const CoordinateConverters& conv, const AffineXf3f* xf )
{
    std::array<PreciseVertCoords, 4> vs;
    vs[0].id = m.topology.org( e );
    vs[1].id = m.topology.dest( e );
    vs[2].id = m.topology.dest( m.topology.next( e ) );        // apex of the left face
    vs[3].id = m.topology.dest( m.topology.next( e.sym() ) );  // apex of the right face
    for ( auto& v : vs )
    {
        const auto& q = m.points[v.id];
        v.pt = conv.toInt( xf ? ( *xf )( q ) : q );
    }
    // the left face is oriented so that its right-hand normal looks outside, and orient3d is true
    // when the apex of the right face is behind it, which makes the edge convex
    return orient3d( vs );
}

} //anonymous namespace

bool isNonIntersectingInsidePrecise( const Mesh& a, FaceId aFace, const MeshPart& b,
    const CoordinateConverters& conv, int aVertShift, int bVertShift,
    const AffineXf3f* xfA, const AffineXf3f* xfB )
{
    if ( !aFace )
        return true; //consider empty mesh always inside

    // only mesh vertices have the exact integer coordinates and the ids that the precise predicates need,
    // so a vertex of aFace is taken as the probe point
    VertId aVerts[3];
    a.topology.getTriVerts( aFace, aVerts[0], aVerts[1], aVerts[2] );
    const VertId aVert = aVerts[0];
    auto aPoint = a.points[aVert];
    if ( xfA )
        aPoint = ( *xfA )( aPoint );
    const auto proj = findProjection( aPoint, b, FLT_MAX, xfB );
    if ( !proj )
        return false; //no projection on b at all

    PreciseVertCoords probe;
    probe.id = VertId( int( aVert ) + aVertShift );
    probe.pt = conv.toInt( aPoint );
    const auto& btopo = b.mesh.topology;

    // the plane of one triangle decides only if the projection is strictly inside that triangle;
    // otherwise the probe point is in the normal cone of the edge or the vertex it projects on,
    // where the planes of the incident faces can disagree about the side the probe point is on
    if ( auto bVert = proj.mtp.inVertex( btopo ) )
    {
        std::optional<bool> behind;
        for ( EdgeId e : orgRing( btopo, bVert ) )
        {
            const auto l = btopo.left( e );
            if ( !l )
                continue;
            const bool cur = isBehindFacePrecise( b.mesh, l, probe, conv, bVertShift, xfB );
            if ( !behind )
                behind = cur;
            else if ( *behind != cur )
            {
                // the vertex is supported by a plane, and its normal cone looks outside of b if from
                // outside and inside if from inside, which the planes of the faces cannot tell apart;
                // the pseudonormal behind the sign of the distance can, being unreliable at the zero distance only
                AffineXf3f b2a;
                if ( xfA )
                    b2a = xfA->inverse();
                if ( xfB )
                    b2a = b2a * ( *xfB );
                return isNonIntersectingInside( a, aFace, b, ( xfA || xfB ) ? &b2a : nullptr );
            }
        }
        if ( behind )
            return *behind;
    }
    else if ( auto bEdgePoint = proj.mtp.onEdge( btopo ) )
    {
        const auto l = btopo.left( bEdgePoint.e );
        const auto r = btopo.right( bEdgePoint.e );
        if ( l && r )
        {
            const bool lBehind = isBehindFacePrecise( b.mesh, l, probe, conv, bVertShift, xfB );
            if ( lBehind == isBehindFacePrecise( b.mesh, r, probe, conv, bVertShift, xfB ) )
                return lBehind;
            // the faces disagree, so the edge is not flat, and the both wedges of an edge
            // cannot be non-empty: the probe point is outside of a convex edge and inside of a concave one
            return !isConvexEdgePrecise( b.mesh, bEdgePoint.e, conv, xfB );
        }
    }

    return isBehindFacePrecise( b.mesh, proj.proj.face, probe, conv, bVertShift, xfB );
}

} //namespace MR
