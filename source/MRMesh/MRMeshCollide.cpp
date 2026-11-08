#include "MRMeshCollide.h"
#include "MRAABBTreeWideTraversal.hpp"
#include "MRMesh.h"
#include "MRParallelFor.h"
#include "MRTriangleIntersection.h"
#include "MREnums.h"
#include "MRTimer.h"
#include "MRTriMath.h"
#include "MRTriDist.h"
#include "MRExpected.h"
#include "MRMeshProject.h"
#include "MRPrecisePredicates3.h"
#include "MRRingIterator.h"

#include <atomic>
#include <optional>
#include <thread>

namespace MR
{

std::vector<FaceFace> findCollidingTriangles( const MeshPart & a, const MeshPart & b, const AffineXf3f * rigidB2A, bool firstIntersectionOnly )
{
    MR_TIMER;

    std::vector<FaceFace> res;
    const AABBTreeWide & aTree = a.mesh.getAABBTreeWide();
    const AABBTreeWide & bTree = b.mesh.getAABBTreeWide();
    if ( aTree.nodes().empty() || bTree.nodes().empty() )
        return res;

    NodeBitSet aNodes, bNodes;
    if ( a.region )
        aNodes = aTree.getNodesFromFaces( *a.region );
    if ( b.region )
        bNodes = bTree.getNodesFromFaces( *b.region );

    std::optional<LazyTransformedBoxes<NodeId>> bNodeBoxes;
    std::optional<LazyTransformedBoxes<size_t>> bLeafBoxes;
    if ( rigidB2A )
    {
        bNodeBoxes.emplace( bTree.nodes().size(), *rigidB2A );
        bLeafBoxes.emplace( bTree.leafBoxes().size(), *rigidB2A );
    }
    auto bNodeBox = [&]( NodeId n ) { return bNodeBoxes ? bNodeBoxes->get( n, bTree.nodes()[n].box ) : bTree.nodes()[n].box; };
    auto bLeafBox = [&]( int i ) { return bLeafBoxes ? bLeafBoxes->get( i, bTree.leafBoxes()[i] ) : bTree.leafBoxes()[i]; };
    auto nodesIntersect = [&]( NodeId an, NodeId bn )
    {
        return ( !a.region || aNodes.test( an ) ) && ( !b.region || bNodes.test( bn ) ) && aTree.nodes()[an].box.intersects( bNodeBox( bn ) );
    };

    // the pairs of triangles with intersecting boxes, as their indices in leafTris()
    std::vector<std::pair<int, int>> tris;
    Vector<WideNodePair> stack;
    if ( nodesIntersect( NodeId{ 0 }, NodeId{ 0 } ) )
        stack.push_back( { NodeId{ 0 }, NodeId{ 0 } } );
    while ( !stack.empty() )
    {
        const auto s = stack.back();
        stack.pop_back();
        if ( splitWidePair( aTree, bTree, s, nodesIntersect, [&stack]( const WideNodePair & c ) { stack.push_back( c ); } ) )
            continue;
        forWideLeafTris( aTree, bTree, s,
            [&]( int i, NodeId bn ) { return aTree.leafBoxes()[i].intersects( bNodeBox( bn ) ); },
            [&]( int i, int j ) { return aTree.leafBoxes()[i].intersects( bLeafBox( j ) ); },
            [&]( int i, int j )
            {
                if ( ( !a.region || a.region->test( aTree.leafTris()[i].face ) ) && ( !b.region || b.region->test( bTree.leafTris()[j].face ) ) )
                    tris.emplace_back( i, j );
                return true;
            } );
    }

    res.resize( tris.size() );
    std::atomic<int> firstIntersection{ (int)res.size() };
    ParallelFor( res, [&] ( size_t i )
    {
        const auto & at = aTree.leafTris()[tris[i].first];
        const auto & bt = bTree.leafTris()[tris[i].second];
        res[i] = FaceFace( at.face, bt.face );
        int knownIntersection = firstIntersection.load( std::memory_order_relaxed );
        if ( firstIntersectionOnly && knownIntersection < i )
            return;
        Vector3f av[3], bv[3];
        for ( int j = 0; j < 3; ++j )
        {
            av[j] = a.mesh.points[at.verts[j]];
            bv[j] = b.mesh.points[bt.verts[j]];
            if ( rigidB2A )
                bv[j] = (*rigidB2A)( bv[j] );
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
    const AABBTreeWide & tree = mp.mesh.getAABBTreeWide();
    if ( tree.nodes().empty() )
        return false;

    auto nodesIntersect = [&tree]( NodeId x, NodeId y ) { return tree.nodes()[x].box.intersects( tree.nodes()[y].box ); };
    auto split = [&]( const WideNodePair & s, auto && push ) { return splitSelfWidePair( tree, s, nodesIntersect, push ); };
    // 8 levels -> at most 4^8 subtasks
    const auto subtasks = makeWideSubtasks( { { NodeId{ 0 }, NodeId{ 0 } } }, 8, split );
    if ( !reportProgress( cb, 0.08f ) )
        return unexpectedOperationCanceled();

    auto sb = subprogress( cb, 0.08f, 0.92f );

    std::vector<std::vector<FaceFace>> subtaskRes( subtasks.size() );

    auto mainThreadId = std::this_thread::get_id();
    std::atomic<bool> keepGoing{ true };
    std::atomic<size_t> numDone;
    // checks subtasks in parallel
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, subtasks.size() ),
        [&]( const tbb::blocked_range<size_t>& range )
    {
        Vector<WideNodePair> mySubtasks;
        for ( auto is = range.begin(); is < range.end(); ++is )
        {
            if ( sb && !keepGoing.load( std::memory_order_relaxed ) )
                break;

            std::vector<FaceFace> myRes;
            // returns false to stop the search
            auto onTris = [&tree, &mp, &myRes, regionMap, outCollidingPairs, &keepGoing, touchIsIntersection]( int ia, int ib )
                {
                    const auto aFace = tree.leafTris()[ia].face;
                    if ( mp.region && !mp.region->test( aFace ) )
                        return true;
                    const auto bFace = tree.leafTris()[ib].face;
                    if ( mp.region && !mp.region->test( bFace ) )
                        return true;
                    if ( regionMap && ( *regionMap )[aFace] != ( *regionMap )[bFace] )
                        return true;

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
                            return true; // triangles sharing an edge may only touch one another

                        const auto na = normal( ap );
                        const auto nb = normal( bp );
                        constexpr auto epsSq = sqr( 1e-5 ); // angle must less than 5.73e-4 deg to consider two faces coplanar
                        if ( cross( na, nb ).lengthSq() > epsSq )
                            return true; // triangles are not coplanar

                        if ( dot( na, nb ) > 0 )
                            return true; // triangles are coplanar, but same-oriented, so they are separated by the shared edge

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
                                return true;
                            // else not touching
                        }
                    }
                    else if ( auto td = findTriTriDistance( { ap[0], ap[1], ap[2] }, { bp[0], bp[1], bp[2] }, { .upDistLimitSq = 0, .upLimitCheck = touchIsIntersection ? UpLimitCheck::Greater : UpLimitCheck::GreaterOrEqual } );
                        td.distSq > 0 || ( !touchIsIntersection && !td.overlap ) )
                    {
                        return true;
                    }
                    myRes.emplace_back( aFace, bFace );
                    if ( !outCollidingPairs )
                    {
                        keepGoing.store( false, std::memory_order_relaxed );
                        return false;
                    }
                    return true;
                };

            mySubtasks.push_back( subtasks[is] );
            while ( !mySubtasks.empty() )
            {
                const auto s = mySubtasks.back();
                mySubtasks.pop_back();
                if ( split( s, [&mySubtasks]( const WideNodePair & c ) { mySubtasks.push_back( c ); } ) )
                    continue;
                if ( !forWideLeafTris<true>( tree, tree, s,
                    [&tree]( int i, NodeId n ) { return tree.leafBoxes()[i].intersects( tree.nodes()[n].box ); },
                    [&tree]( int i, int j ) { return tree.leafBoxes()[i].intersects( tree.leafBoxes()[j] ); },
                    onTris ) )
                {
                    mySubtasks.clear();
                    break;
                }
            }

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

    auto sd = findSignedDistance( aPoint, b );
    // a point projecting on the boundary of b is beyond the edge of an open surface, so it is outside whatever side it is on
    return sd && sd->dist < 0 && !sd->mtp.isBd( b.mesh.topology, b.region );
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
    if ( !proj || proj.mtp.isBd( b.mesh.topology, b.region ) )
        return false; //no projection on b at all, or beyond its boundary

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
