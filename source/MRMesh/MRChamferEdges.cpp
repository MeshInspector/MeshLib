#include "MRChamferEdges.h"
#include "MRMesh.h"
#include "MRRegionBoundary.h"
#include "MRSurfaceDistance.h"
#include "MRExtractIsolines.h"
#include "MROneMeshContours.h"
#include "MRContoursCut.h"
#include "MRFillContour.h"
#include "MRMeshSubdivide.h"
#include "MRMeshComponents.h"
#include "MRAABBTreePolyline.h"
#include "MRPolylineProject.h"
#include "MRBitSetParallelFor.h"
#include "MRRingIterator.h"
#include "MRLine.h"
#include "MRLineSegm.h"
#include "MRTimer.h"
#include <climits>
#include <cmath>
#include <limits>
#include <optional>

namespace MR
{

Expected<FaceBitSet> chamferEdges( Mesh & mesh, const UndirectedEdgeBitSet & edges, float distance, const ProgressCallback & cb )
{
    MR_TIMER;
    if ( !( distance > 0 ) )
        return unexpected( "Chamfer distance must be positive" );

    auto & tp = mesh.topology;
    if ( edges.none() )
        return unexpected( "No edges to chamfer" );
    for ( auto ue : edges )
        if ( !tp.left( ue ) || !tp.right( ue ) )
            return unexpected( "Chamfered edges must be inner edges of the mesh" );
    for ( auto v : getIncidentVerts( tp, edges ) )
    {
        int n = 0;
        for ( auto e : orgRing( tp, v ) )
            n += edges.test( e.undirected() );
        if ( n != 2 )
            return unexpected( "Chamfered edges must form disjoint closed loops" );
    }

    // original edges are straight segments, the creases between their chamfers are found from them
    Vector<LineSegm3f, UndirectedEdgeId> origSegm( tp.undirectedEdgeSize() );
    Vector<UndirectedEdgeId, UndirectedEdgeId> origEdge( tp.undirectedEdgeSize() );
    for ( auto ue : edges )
    {
        origSegm[ue] = mesh.edgeSegment( ue );
        origEdge[ue] = ue;
    }

    auto selEdges = edges;
    // vertices at surface distance less than 2*distance from the edges
    auto getNearVerts = [&]
    {
        const auto nearDist = computeSurfaceDistances( mesh, getIncidentVerts( tp, selEdges ), 2 * distance );
        VertBitSet nearVerts( nearDist.size() );
        BitSetParallelFor( tp.getValidVerts(), [&]( VertId v )
        {
            if ( nearDist[v] < 2 * distance )
                nearVerts.set( v );
        } );
        return nearVerts;
    };
    auto nearFaces = getIncidentFaces( tp, getNearVerts() );

    // halve the edge length on each pass and shrink the region again, not to subdivide far parts of big triangles
    float maxLen = 0;
    for ( auto ue : getIncidentEdges( tp, nearFaces ) )
        maxLen = std::max( maxLen, mesh.edgeLength( ue ) );
    const int numPasses = std::max( 1, (int)std::ceil( std::log2( maxLen / distance ) ) );

    SubdivideSettings ss;
    ss.maxEdgeSplits = INT_MAX;
    ss.maxDeviationAfterFlip = distance * std::numeric_limits<float>::epsilon();
    ss.region = &nearFaces;
    ss.notFlippable = &selEdges;
    ss.onEdgeSplit = [&]( EdgeId e1, EdgeId e )
    {
        if ( contains( selEdges, e.undirected() ) )
            origEdge.autoResizeSet( e1.undirected(), origEdge[e.undirected()] );
    };
    for ( int i = 0; i < numPasses; ++i )
    {
        if ( i > 0 )
            nearFaces = getIncidentFaces( tp, getNearVerts() );
        ss.maxEdgeLen = distance * std::exp2( float( numPasses - 1 - i ) );
        ss.progressCallback = subprogress( cb, 0.4f * i / numPasses, 0.4f * ( i + 1 ) / numPasses );
        subdivideMesh( mesh, ss );
        if ( !reportProgress( cb, 0.4f * ( i + 1 ) / numPasses ) )
            return unexpectedOperationCanceled();
    }

    const AABBTreePolyline3 edgesTree( mesh, selEdges );
    const float maxDistSq = sqr( 2 * distance );
    auto nearestOrigAt = [&]( const Vector3f & p )
    {
        const auto proj = findProjectionOnMeshEdges( p, mesh, edgesTree, maxDistSq );
        return proj.valid() ? origEdge[proj.line] : UndirectedEdgeId{};
    };

    // split edges where the nearest original edge changes, so that the creases between chamfers of neighbor edges are mesh edges
    auto nearVerts = getNearVerts();
    Vector<UndirectedEdgeId, VertId> nearestOrig( tp.vertSize() );
    BitSetParallelFor( nearVerts, [&]( VertId v )
    {
        nearestOrig[v] = nearestOrigAt( mesh.points[v] );
    } );
    // positive if (p) is closer to original edge (j) than to (i)
    auto closerToSecond = [&]( const Vector3f & p, UndirectedEdgeId i, UndirectedEdgeId j )
    {
        return ( p - closestPointOnLineSegm( p, origSegm[i] ) ).length() - ( p - closestPointOnLineSegm( p, origSegm[j] ) ).length();
    };
    const float creaseTol = 1e-4f * distance;
    std::vector<EdgeId> crossingEdges;
    for ( auto ue : getIncidentEdges( tp, getIncidentFaces( tp, nearVerts ) ) )
    {
        const auto i = nearestOrig[tp.org( ue )];
        const auto j = nearestOrig[tp.dest( ue )];
        if ( i && j && i != j && !selEdges.test( ue ) )
            crossingEdges.push_back( ue );
    }
    // one edge can cross several creases if the original edges are shorter than mesh edges
    for ( EdgeId e : crossingEdges )
    {
        auto i = nearestOrig[tp.org( e )];
        const auto last = nearestOrig[tp.dest( e )];
        while ( i != last )
        {
            const auto po = mesh.orgPnt( e );
            auto pEnd = mesh.destPnt( e );
            auto j = last;
            std::optional<Vector3f> crease;
            for ( int attempt = 0; attempt < 16; ++attempt )
            {
                const float go = closerToSecond( po, i, j );
                const float ge = closerToSecond( pEnd, i, j );
                if ( !( go < -creaseTol && ge > creaseTol ) )
                    break;
                // the difference of distances is not linear along the edge near the ends of original edges
                float t0 = 0, t1 = 1;
                for ( int iter = 0; iter < 24; ++iter )
                {
                    const float t = 0.5f * ( t0 + t1 );
                    ( closerToSecond( po + ( pEnd - po ) * t, i, j ) < 0 ? t0 : t1 ) = t;
                }
                const auto x = po + ( pEnd - po ) * ( 0.5f * ( t0 + t1 ) );
                const auto k = nearestOrigAt( x );
                if ( !k )
                    break;
                if ( k == i || k == j )
                {
                    crease = x;
                    break;
                }
                // another original edge is closer at the crossing, so search the crease with it on the first part of the edge
                j = k;
                pEnd = x;
            }
            if ( !crease )
                break;
            mesh.splitEdge( e, *crease ); // now (e) starts at the new vertex
            i = j;
        }
    }
    if ( !reportProgress( cb, 0.45f ) )
        return unexpectedOperationCanceled();

    // the borders are at the given straight distance from the edges, which is used again to move the vertices
    nearVerts = getNearVerts();
    VertScalars dist( tp.vertSize(), FLT_MAX );
    BitSetParallelFor( nearVerts, [&]( VertId v )
    {
        dist[v] = std::sqrt( findProjectionOnMeshEdges( mesh.points[v], mesh, edgesTree, maxDistSq ).distSq );
    } );
    const auto isoLines = extractIsolines( tp, dist, distance );
    if ( !reportProgress( cb, 0.5f ) )
        return unexpectedOperationCanceled();

    const auto cutRes = cutMesh( mesh, convertSurfacePathsToMeshContours( mesh, isoLines ) );
    if ( cutRes.fbsWithContourIntersections.any() )
        return unexpected( "Chamfer borders intersect each other" );
    if ( !reportProgress( cb, 0.7f ) )
        return unexpectedOperationCanceled();

    auto strip = fillContourLeft( tp, cutRes.resultCut );

    // the edges split the strip on sides, each side is bounded by its own part of the strip boundary
    const auto sideMap = MeshComponents::getAllComponentsMap( { mesh, &strip }, MeshComponents::FaceIncidence::PerEdge, &selEdges );
    const auto & side = sideMap.first;
    const int numSides = sideMap.second;
    for ( auto ue : selEdges )
        if ( side[tp.left( ue )] == side[tp.right( ue )] )
            return unexpected( "Chamfer strip does not have two sides" );

    std::vector<UndirectedEdgeBitSet> sideBorders( numSides, UndirectedEdgeBitSet( tp.undirectedEdgeSize() ) );
    for ( auto ue : getIncidentEdges( tp, strip ) )
    {
        const auto l = tp.left( ue );
        const auto r = tp.right( ue );
        const bool inL = contains( strip, l );
        if ( inL != contains( strip, r ) )
            sideBorders[side[inL ? l : r]].set( ue );
    }

    std::vector<AABBTreePolyline3> sideBorderTrees;
    sideBorderTrees.reserve( numSides );
    for ( const auto & b : sideBorders )
    {
        if ( b.none() )
            return unexpected( "Chamfer distance is too large" );
        sideBorderTrees.emplace_back( mesh, b );
    }
    if ( !reportProgress( cb, 0.8f ) )
        return unexpectedOperationCanceled();

    // vertex (v) at distance (s) from the edges goes on segment (a,b): (a) is the closest border point to (v) on its side,
    // the ray from (a) through (v) reaches the edges at (onEdge), and (b) is the closest border point to it on the other side
    const auto edgeVerts = getIncidentVerts( tp, selEdges );
    auto newPoints = mesh.points;
    if ( !BitSetParallelFor( getInnerVerts( tp, strip ), [&]( VertId v )
    {
        const auto & pt = mesh.points[v];
        const auto proj = findProjectionOnMeshEdges( pt, mesh, edgesTree, maxDistSq );
        if ( !proj.valid() )
            return;
        const EdgeId e = proj.line;
        auto leftSide = side[tp.left( e )];
        auto rightSide = side[tp.right( e )];
        if ( edgeVerts.test( v ) )
        {
            const auto a = findProjectionOnMeshEdges( pt, mesh, sideBorderTrees[leftSide], maxDistSq );
            const auto b = findProjectionOnMeshEdges( pt, mesh, sideBorderTrees[rightSide], maxDistSq );
            if ( a.valid() && b.valid() )
                newPoints[v] = 0.5f * ( a.point + b.point );
            return;
        }
        const auto ownSide = side[tp.left( tp.edgeWithOrg( v ) )];
        if ( ownSide == rightSide )
            std::swap( leftSide, rightSide );
        else if ( ownSide != leftSide )
            return;
        const auto a = findProjectionOnMeshEdges( pt, mesh, sideBorderTrees[leftSide], maxDistSq );
        if ( !a.valid() )
            return;
        const float s = std::sqrt( proj.distSq );
        if ( s >= distance )
        {
            newPoints[v] = a.point;
            return;
        }
        const auto onEdge = a.point + ( pt - a.point ) * ( distance / ( distance - s ) );
        const auto b = findProjectionOnMeshEdges( onEdge, mesh, sideBorderTrees[rightSide], maxDistSq );
        if ( b.valid() )
            newPoints[v] = a.point + ( b.point - a.point ) * ( ( distance - s ) / ( 2 * distance ) );
    }, subprogress( cb, 0.8f, 1.0f ) ) )
        return unexpectedOperationCanceled();

    mesh.points = std::move( newPoints );
    mesh.invalidateCaches();
    return strip;
}

} // namespace MR
