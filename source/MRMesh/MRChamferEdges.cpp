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
#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <optional>
#include <tuple>

namespace MR
{

namespace
{

Expected<void> checkChamferInput( const MeshTopology & tp, const UndirectedEdgeBitSet & edges, float distance )
{
    if ( !( distance > 0 ) )
        return unexpected( "Chamfer distance must be positive" );
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
        if ( n != 2 && !( n == 1 && tp.isBdVertex( v ) ) )
            return unexpected( "Chamfered edges must form disjoint closed loops or chains ending on the mesh boundary" );
    }
    return {};
}

/// the chamfered edges after subdivision, each remembers the original edge it is a part of
struct SubdividedEdges
{
    UndirectedEdgeBitSet edges;
    Vector<UndirectedEdgeId, UndirectedEdgeId> origEdge;
    /// original edges are straight segments, the creases between their chamfers are found from them
    Vector<LineSegm3f, UndirectedEdgeId> origSegm;

    SubdividedEdges( const Mesh & mesh, const UndirectedEdgeBitSet & e ) : edges( e ),
        origEdge( mesh.topology.undirectedEdgeSize() ), origSegm( mesh.topology.undirectedEdgeSize() )
    {
        for ( auto ue : edges )
        {
            origEdge[ue] = ue;
            origSegm[ue] = mesh.edgeSegment( ue );
        }
    }
};

/// vertices at surface distance less than (maxDist) from the edges
VertBitSet getNearVerts( const Mesh & mesh, const UndirectedEdgeBitSet & edges, float maxDist )
{
    const auto dist = computeSurfaceDistances( mesh, getIncidentVerts( mesh.topology, edges ), maxDist );
    VertBitSet res( dist.size() );
    BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        if ( dist[v] < maxDist )
            res.set( v );
    } );
    return res;
}

/// subdivides the triangles near the edges till their edges are not longer than (distance)
bool subdivideNearEdges( Mesh & mesh, SubdividedEdges & se, float distance, const ProgressCallback & cb )
{
    MR_TIMER;
    const auto & tp = mesh.topology;
    auto nearFaces = getIncidentFaces( tp, getNearVerts( mesh, se.edges, 2 * distance ) );

    // halve the edge length on each pass and shrink the region again, not to subdivide far parts of big triangles
    float maxLen = 0;
    for ( auto ue : getIncidentEdges( tp, nearFaces ) )
        maxLen = std::max( maxLen, mesh.edgeLength( ue ) );
    const int numPasses = std::max( 1, (int)std::ceil( std::log2( maxLen / distance ) ) );

    SubdivideSettings ss;
    ss.maxEdgeSplits = INT_MAX;
    ss.maxDeviationAfterFlip = distance * std::numeric_limits<float>::epsilon();
    ss.region = &nearFaces;
    ss.notFlippable = &se.edges;
    ss.onEdgeSplit = [&]( EdgeId e1, EdgeId e )
    {
        if ( contains( se.edges, e.undirected() ) )
            se.origEdge.autoResizeSet( e1.undirected(), se.origEdge[e.undirected()] );
    };
    for ( int i = 0; i < numPasses; ++i )
    {
        if ( i > 0 )
            nearFaces = getIncidentFaces( tp, getNearVerts( mesh, se.edges, 2 * distance ) );
        ss.maxEdgeLen = distance * std::exp2( float( numPasses - 1 - i ) );
        ss.progressCallback = subprogress( cb, float( i ) / numPasses, float( i + 1 ) / numPasses );
        subdivideMesh( mesh, ss );
        if ( !reportProgress( cb, float( i + 1 ) / numPasses ) )
            return false;
    }

    // a triangle with all vertices on the edges would collapse on the middle line of the chamfer
    const auto edgeVerts = getIncidentVerts( tp, se.edges );
    std::vector<EdgeId> chords;
    for ( auto ue : getIncidentEdges( tp, getIncidentFaces( tp, edgeVerts ) ) )
        if ( !se.edges.test( ue ) && edgeVerts.test( tp.org( ue ) ) && edgeVerts.test( tp.dest( ue ) ) )
            chords.push_back( ue );
    for ( auto e : chords )
        mesh.splitEdge( e );
    return true;
}

/// splits mesh edges near the chamfered edges where the nearest original edge changes,
/// so that the creases between chamfers of neighbor edges become mesh edges
void splitCreases( Mesh & mesh, const SubdividedEdges & se, const AABBTreePolyline3 & edgesTree, float distance )
{
    MR_TIMER;
    const auto & tp = mesh.topology;
    const float maxDistSq = sqr( 2 * distance );
    auto nearestOrigAt = [&]( const Vector3f & p )
    {
        const auto proj = findProjectionOnMeshEdges( p, mesh, edgesTree, maxDistSq );
        return proj.valid() ? se.origEdge[proj.line] : UndirectedEdgeId{};
    };
    // positive if (p) is closer to original edge (j) than to (i)
    auto closerToSecond = [&]( const Vector3f & p, UndirectedEdgeId i, UndirectedEdgeId j )
    {
        return ( p - closestPointOnLineSegm( p, se.origSegm[i] ) ).length() - ( p - closestPointOnLineSegm( p, se.origSegm[j] ) ).length();
    };

    const auto nearVerts = getNearVerts( mesh, se.edges, 2 * distance );
    Vector<UndirectedEdgeId, VertId> nearestOrig( tp.vertSize() );
    BitSetParallelFor( nearVerts, [&]( VertId v )
    {
        nearestOrig[v] = nearestOrigAt( mesh.points[v] );
    } );
    std::vector<EdgeId> crossingEdges;
    for ( auto ue : getIncidentEdges( tp, getIncidentFaces( tp, nearVerts ) ) )
    {
        const auto i = nearestOrig[tp.org( ue )];
        const auto j = nearestOrig[tp.dest( ue )];
        if ( i && j && i != j && !se.edges.test( ue ) )
            crossingEdges.push_back( ue );
    }

    const float creaseTol = 1e-4f * distance;
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
            float creaseT = 0;
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
                const float t = 0.5f * ( t0 + t1 );
                const auto x = po + ( pEnd - po ) * t;
                const auto k = nearestOrigAt( x );
                if ( !k )
                    break;
                if ( k == i || k == j )
                {
                    crease = x;
                    creaseT = t;
                    break;
                }
                // another original edge is closer at the crossing, so search the crease with it on the first part of the edge
                j = k;
                pEnd = x;
            }
            if ( !crease )
                break;
            // a vertex very close to the crease is used instead of a new one, not to make slivers
            const float snapT = 0.05f * mesh.edgeLength( e ) / ( pEnd - po ).length();
            if ( creaseT > 1 - snapT && j == last )
                break;
            if ( creaseT >= snapT )
                mesh.splitEdge( e, *crease ); // now (e) starts at the new vertex
            i = j;
        }
    }
}

/// the width of the chamfer at each vertex of the edges: (distance), but less than half the gap to other loops or chains,
/// or to the parts of the same loop far along it, so that the chamfer strips do not overlap
VertScalars computeWidths( const Mesh & mesh, const UndirectedEdgeBitSet & edges, const AABBTreePolyline3 & edgesTree, float distance )
{
    MR_TIMER;
    const auto & tp = mesh.topology;
    const auto edgeVerts = getIncidentVerts( tp, edges );

    // walk each loop or chain to find its id and the arc length of each vertex along it
    Vector<int, VertId> part( tp.vertSize(), -1 );
    VertScalars arc( tp.vertSize() );
    std::vector<float> partLength;
    auto nextEdge = [&]( EdgeId e ) // the other edge of the loop at dest( e )
    {
        for ( auto en : orgRing( tp, e.sym() ) )
            if ( en != e.sym() && edges.test( en.undirected() ) )
                return en;
        return EdgeId{};
    };
    auto walk = [&]( EdgeId e )
    {
        const int id = (int)partLength.size();
        float len = 0;
        part[tp.org( e )] = id;
        while ( e && part[tp.dest( e )] < 0 )
        {
            len += mesh.edgeLength( e );
            part[tp.dest( e )] = id;
            arc[tp.dest( e )] = len;
            e = nextEdge( e );
        }
        // the arc length along a chain is not cyclic
        partLength.push_back( e ? len + mesh.edgeLength( e ) : FLT_MAX );
    };
    for ( auto v : edgeVerts ) // chains start at their ends
        if ( part[v] < 0 && tp.isBdVertex( v ) )
            for ( auto e : orgRing( tp, v ) )
                if ( edges.test( e.undirected() ) )
                {
                    walk( e );
                    break;
                }
    for ( auto v : edgeVerts )
        if ( part[v] < 0 )
            for ( auto e : orgRing( tp, v ) )
                if ( edges.test( e.undirected() ) )
                {
                    walk( e );
                    break;
                }

    VertScalars res( tp.vertSize(), distance );
    BitSetParallelFor( edgeVerts, [&]( VertId v )
    {
        const auto & p = mesh.points[v];
        const int id = part[v];
        float gap = FLT_MAX;
        findMeshEdgesInBall( mesh, edgesTree, p, 2 * distance, [&]( UndirectedEdgeId ue, const Vector3f & q, float distSq )
        {
            const auto w = tp.org( ue );
            if ( part[w] == id )
            {
                // corners of the same loop are not gaps, only its parts far along it are
                float along = std::abs( arc[w] - arc[v] );
                along = std::min( along, partLength[id] - along );
                if ( along <= 2 * std::sqrt( distSq ) + 2 * distance )
                    return;
            }
            gap = std::min( gap, ( q - p ).length() );
        } );
        res[v] = std::min( distance, 0.45f * gap );
    } );
    return res;
}

/// the width of the chamfer at point (p) on edge (ue)
float widthAt( const Mesh & mesh, const VertScalars & width, UndirectedEdgeId ue, const Vector3f & p )
{
    const auto o = mesh.topology.org( ue );
    const auto d = mesh.topology.dest( ue );
    const float len = mesh.edgeLength( ue );
    const float t = len > 0 ? std::clamp( ( p - mesh.points[o] ).length() / len, 0.0f, 1.0f ) : 0.0f;
    return ( 1 - t ) * width[o] + t * width[d];
}

/// cuts the mesh along the lines at the chamfer width from the edges
/// \return the triangles closer to the edges than the cut lines
Expected<FaceBitSet> cutAtDistance( Mesh & mesh, const UndirectedEdgeBitSet & edges, const AABBTreePolyline3 & edgesTree,
    const VertScalars & width, float distance )
{
    MR_TIMER;
    const auto & tp = mesh.topology;
    const float maxDistSq = sqr( 2 * distance );
    VertScalars relDist( tp.vertSize(), FLT_MAX );
    BitSetParallelFor( getNearVerts( mesh, edges, 2 * distance ), [&]( VertId v )
    {
        const auto proj = findProjectionOnMeshEdges( mesh.points[v], mesh, edgesTree, maxDistSq );
        if ( proj.valid() )
            relDist[v] = std::sqrt( proj.distSq ) / widthAt( mesh, width, proj.line, proj.point );
    } );

    const auto cutRes = cutMesh( mesh, convertSurfacePathsToMeshContours( mesh, extractIsolines( tp, relDist, 1.0f ) ) );
    if ( cutRes.fbsWithContourIntersections.any() )
        return unexpected( "Chamfer borders intersect each other" );
    return fillContourLeft( tp, cutRes.resultCut );
}

/// the edges divide the strip on sides, and each side is bounded by its own chamfer border
struct StripSides
{
    Face2RegionMap side;
    std::vector<AABBTreePolyline3> borderTrees;
    VertBitSet borderVerts;
};

Expected<StripSides> findStripSides( const Mesh & mesh, const FaceBitSet & strip, const UndirectedEdgeBitSet & edges )
{
    MR_TIMER;
    const auto & tp = mesh.topology;
    StripSides res;
    int numSides = 0;
    std::tie( res.side, numSides ) = MeshComponents::getAllComponentsMap( { mesh, &strip }, MeshComponents::FaceIncidence::PerEdge, &edges );
    for ( auto ue : edges )
        if ( res.side[tp.left( ue )] == res.side[tp.right( ue )] )
            return unexpected( "Chamfer strip does not have two sides" );

    // the mesh boundary is not a chamfer border
    std::vector<UndirectedEdgeBitSet> borders( numSides, UndirectedEdgeBitSet( tp.undirectedEdgeSize() ) );
    for ( auto ue : getIncidentEdges( tp, strip ) )
    {
        const auto l = tp.left( ue );
        const auto r = tp.right( ue );
        const bool inL = contains( strip, l );
        if ( l && r && inL != contains( strip, r ) )
            borders[res.side[inL ? l : r]].set( ue );
    }

    res.borderTrees.reserve( numSides );
    res.borderVerts.resize( tp.vertSize() );
    for ( const auto & b : borders )
    {
        if ( b.none() )
            return unexpected( "Chamfer distance is too large" );
        res.borderTrees.emplace_back( mesh, b );
        res.borderVerts |= getIncidentVerts( tp, b );
    }
    return res;
}

/// moves all strip vertices except for the borders on the chamfer surface
bool moveStripVerts( Mesh & mesh, const FaceBitSet & strip, const UndirectedEdgeBitSet & edges, const AABBTreePolyline3 & edgesTree,
    const VertScalars & width, const StripSides & sides, float distance, const ProgressCallback & cb )
{
    MR_TIMER;
    const auto & tp = mesh.topology;
    const float maxDistSq = sqr( 2 * distance );
    const auto edgeVerts = getIncidentVerts( tp, edges );
    auto newPoints = mesh.points;

    // vertex (v) at relative distance (s) from the edges goes on segment (a,b): (a) is the closest border point to (v) on its side,
    // the ray from (a) through (v) reaches the edges at (onEdge), and (b) is the closest border point to it on the other side
    if ( !BitSetParallelFor( getIncidentVerts( tp, strip ) - sides.borderVerts, [&]( VertId v )
    {
        const auto & pt = mesh.points[v];
        const auto proj = findProjectionOnMeshEdges( pt, mesh, edgesTree, maxDistSq );
        if ( !proj.valid() )
            return;
        const EdgeId e = proj.line;
        auto leftSide = sides.side[tp.left( e )];
        auto rightSide = sides.side[tp.right( e )];
        if ( edgeVerts.test( v ) )
        {
            const auto a = findProjectionOnMeshEdges( pt, mesh, sides.borderTrees[leftSide], maxDistSq );
            const auto b = findProjectionOnMeshEdges( pt, mesh, sides.borderTrees[rightSide], maxDistSq );
            if ( a.valid() && b.valid() )
                newPoints[v] = 0.5f * ( a.point + b.point );
            return;
        }
        FaceId ownFace;
        for ( auto ev : orgRing( tp, v ) )
            if ( auto f = tp.left( ev ); contains( strip, f ) )
                ownFace = f;
        const auto ownSide = sides.side[ownFace];
        if ( ownSide == rightSide )
            std::swap( leftSide, rightSide );
        else if ( ownSide != leftSide )
            return;
        const auto a = findProjectionOnMeshEdges( pt, mesh, sides.borderTrees[leftSide], maxDistSq );
        if ( !a.valid() )
            return;
        const float s = std::sqrt( proj.distSq ) / widthAt( mesh, width, proj.line, proj.point );
        if ( s >= 1 )
        {
            newPoints[v] = a.point;
            return;
        }
        const auto onEdge = a.point + ( pt - a.point ) / ( 1 - s );
        const auto b = findProjectionOnMeshEdges( onEdge, mesh, sides.borderTrees[rightSide], maxDistSq );
        if ( b.valid() )
            newPoints[v] = a.point + ( b.point - a.point ) * ( 0.5f * ( 1 - s ) );
    }, cb ) )
        return false;

    mesh.points = std::move( newPoints );
    mesh.invalidateCaches();
    return true;
}

} // anonymous namespace

Expected<FaceBitSet> chamferEdges( Mesh & mesh, const UndirectedEdgeBitSet & edges, float distance, const ProgressCallback & cb )
{
    MR_TIMER;
    if ( auto c = checkChamferInput( mesh.topology, edges, distance ); !c )
        return unexpected( std::move( c.error() ) );

    SubdividedEdges se( mesh, edges );
    if ( !subdivideNearEdges( mesh, se, distance, subprogress( cb, 0.0f, 0.4f ) ) )
        return unexpectedOperationCanceled();

    const AABBTreePolyline3 edgesTree( mesh, se.edges );
    splitCreases( mesh, se, edgesTree, distance );
    if ( !reportProgress( cb, 0.45f ) )
        return unexpectedOperationCanceled();

    const auto width = computeWidths( mesh, se.edges, edgesTree, distance );
    auto strip = cutAtDistance( mesh, se.edges, edgesTree, width, distance );
    if ( !strip )
        return strip;
    if ( !reportProgress( cb, 0.7f ) )
        return unexpectedOperationCanceled();

    const auto sides = findStripSides( mesh, *strip, se.edges );
    if ( !sides )
        return unexpected( sides.error() );
    if ( !reportProgress( cb, 0.8f ) )
        return unexpectedOperationCanceled();

    if ( !moveStripVerts( mesh, *strip, se.edges, edgesTree, width, *sides, distance, subprogress( cb, 0.8f, 1.0f ) ) )
        return unexpectedOperationCanceled();
    return strip;
}

} // namespace MR
