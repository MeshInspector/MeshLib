#include "MRContoursCut.h"
#include "MRMesh.h"
#include "MRPolyline.h"
#include "MRTriangleIntersection.h"
#include "MRTriMath.h"
#include "MRMeshTopology.h"
#include "MRMeshDelone.h"
#include "MR2DContoursTriangulation.h"
#include "MRCutTrianglePredicates.h"
#include "MRPrecisePredicates3Poly.h"
#include "MRRingIterator.h"
#include "MRMeshFillHole.h"
#include "MRTimer.h"
#include "MRMeshBuilder.h"
#include "MRBox.h"
#include "MRFillContour.h"
#include "MRMeshComponents.h"
#include "MRSurfaceDistance.h"
#include "MRExtractIsolines.h"
#include "MRParallelFor.h"
#include "MRMeshIntersect.h"
#include "MRSurfacePath.h"
#include "MRphmap.h"
#include "MRFillContourByGraphCut.h"
#include "MREdgeMetric.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRTBB.h"
#include <numeric>
#include <span>

namespace MR
{

namespace
{

class ContourTag;
class IntersectionTag;
using IntersectionId = Id<IntersectionTag>;
using ContourId = Id<ContourTag>;

struct IntersectionData
{
    ContourId contourId;
    IntersectionId intersectionId;
};

struct EdgeIntersectionData
{
    IntersectionData interOnEdge;
    VertId newVert;
    EdgeId orgEdgeInLeftTri;
    int beforeSortIndex{ 0 }; // useful for next sort
};

using EdgeData = std::vector<EdgeIntersectionData>;

// the faces of the old mesh on both sides of a path edge: the same face for a path edge inside it,
// the faces of an edge of the old mesh for a path edge along it
struct PathsEdgeIndex
{
    FaceId left;
    FaceId right;
};

using PathsOldFaces = std::vector<std::vector<PathsEdgeIndex>>; // [contour][edge]
using EdgeDataMap = ParallelHashMap<UndirectedEdgeId, EdgeData>;
struct PreCutResult
{
    EdgeDataMap edgeData;
    std::vector<EdgePath> paths;
    PathsOldFaces oldEdgesInfo;
};

/// prepares precise-coordinates for a vertex of mesh A or B as the boolean's predicates take them:
/// the ids of B shifted by meshAVertsNum, its points moved by rigidB2A in float before the conversion
PreciseVertCoords preciseVert( VertId v, const Mesh& mesh, bool isA, const SortIntersectionsData& sortData )
{
    if ( isA )
        return { v, sortData.converter( mesh.points[v] ) };
    const auto& p = mesh.points[v];
    return { v + int( sortData.meshAVertsNum ), sortData.converter( sortData.rigidB2A ? ( *sortData.rigidB2A )( p ) : p ) };
}

/// prepares precise-coordinates for a vertex from the other mesh
PreciseVertCoords preciseOtherVert( VertId v, const SortIntersectionsData& sortData )
{
    return preciseVert( v, sortData.otherMesh, sortData.isOtherA, sortData );
}

/// prepares precise-coordinates for a vertex of the mesh being cut
PreciseVertCoords preciseOwnVert( VertId v, const Mesh& mesh, const SortIntersectionsData& sortData )
{
    return preciseVert( v, mesh, !sortData.isOtherA, sortData );
}

/// prepares precise-coordinates for the corners of a triangle of the mesh being cut
std::array<PreciseVertCoords, 3> preciseOwnTri( const ThreeVertIds& t, const Mesh& mesh, const SortIntersectionsData& sortData )
{
    return { preciseOwnVert( t[0], mesh, sortData ), preciseOwnVert( t[1], mesh, sortData ), preciseOwnVert( t[2], mesh, sortData ) };
}

/// the exact definition of intersection i of contour c inside a triangle of the mesh with given edges, counterclockwise,
/// and verts[k] = org( edges[k] ) before the cut: a corner, a side of the triangle crossing a triangle of the other mesh,
/// or an edge of the other mesh crossing the triangle
PreciseCutPoint preciseCutPoint( const Mesh& mesh, const OneMeshContours& contours, int c, int i,
    const std::array<EdgeId, 3>& edges, const ThreeVertIds& verts, const SortIntersectionsData& sortData )
{
    PreciseCutPoint res;
    const auto& prim = contours[c].intersections[i].primitiveId;
    const auto& et = sortData.contours[c][i];
    const auto& otherTopology = sortData.otherMesh.topology;
    if ( const auto pe = std::get_if<EdgeId>( &prim ) )
    {
        for ( int k = 0; k < 3; ++k )
            if ( edges[k].undirected() == pe->undirected() )
                res.seg = { preciseOwnVert( verts[k], mesh, sortData ), preciseOwnVert( verts[( k + 1 ) % 3], mesh, sortData ) };
        const auto t = otherTopology.getTriVerts( et.tri() );
        res.tri = { preciseOtherVert( t[0], sortData ), preciseOtherVert( t[1], sortData ), preciseOtherVert( t[2], sortData ) };
    }
    else if ( std::holds_alternative<FaceId>( prim ) )
    {
        // the third vertices of the triangles of the edge tell apart the crossings of a multiple edge
        const EdgeId e = et.edge;
        res.seg = { preciseOtherVert( otherTopology.org( e ), sortData ), preciseOtherVert( otherTopology.dest( e ), sortData ) };
        res.tri = preciseOwnTri( verts, mesh, sortData );
        if ( otherTopology.left( e ) && otherTopology.right( e ) )
            res.wedge = { preciseOtherVert( otherTopology.getLeftTriVerts( e )[2], sortData ),
                preciseOtherVert( otherTopology.getLeftTriVerts( e.sym() )[2], sortData ) };
    }
    else
        res.seg[0] = preciseOwnVert( std::get<VertId>( prim ), mesh, sortData );
    return res;
}

/// the old face that segment i of a contour goes in, from intersection i to i + 1, or invalid for a segment along an edge of the mesh
FaceId segmentFace( const MeshTopology& topology, const std::vector<OneMeshIntersection>& inContour, int i )
{
    const auto& a = inContour[i].primitiveId;
    const auto& b = inContour[i + 1].primitiveId;
    const auto va = std::get_if<VertId>( &a );
    const auto vb = std::get_if<VertId>( &b );
    if ( va && vb )
        return {};
    if ( const auto pe = std::get_if<EdgeId>( va ? &b : &a ) )
    {
        // a path edge leaves an edge point into the left face of the edge and arrives at it from the right face
        const auto e = va ? pe->sym() : *pe;
        const auto v = va ? *va : vb ? *vb : VertId{};
        // precondition: a mesh vertex at the other end is the third vertex of that face, so the contour does not fold back
        // at the edge; a fold puts the path edge in the other face, where Release still gets a valid topology
        // (a mesh vertex that is no corner of these faces is not handled)
        const bool fold = v && ( !topology.left( e ) || topology.dest( topology.next( e ) ) != v );
        assert( !fold );
        return topology.left( fold ? e.sym() : e );
    }
    return std::get<FaceId>( va ? b : a );
}

PreCutResult doPreCutMesh( Mesh& mesh, const OneMeshContours& contours, const SortIntersectionsData* sortData )
{
    MR_TIMER;
    auto& topology = mesh.topology;

    int numVerts = 0;
    int numEdges = 0;
    int intersectedEdges = 0;
    for ( const auto& cont : contours )
    {
        auto size = int( cont.intersections.size() );
        numVerts += size;
        numEdges += 2 * size;
        for ( const auto& in : cont.intersections )
            if ( std::holds_alternative<EdgeId>( in.primitiveId ) )
                ++intersectedEdges;
        if ( cont.closed )
            --numVerts;
    }
    numEdges += 2 * intersectedEdges;
    const auto totalExpectedVerts = topology.vertSize() + numVerts;
    const auto totalExpectedEdges = topology.edgeSize() + numEdges;

    // the fill of the holes adds about 0.5 times as many edges and 0.6 times as many faces as the cut adds edges: reserving for both here,
    // as many as the cut adds for headroom, copies the topology of the whole mesh to the larger buffers once, not again before the fill
    topology.vertReserve( totalExpectedVerts );
    topology.edgeReserve( totalExpectedEdges + numEdges );
    topology.faceReserve( topology.faceSize() + numEdges );
    mesh.points.reserve( totalExpectedVerts );

    PreCutResult res;
    res.paths.resize( contours.size() );
    res.oldEdgesInfo.resize( contours.size() );
    res.edgeData.reserve( size_t( intersectedEdges ) );
    const auto oldEdgesSize = topology.edgeSize();

    // find the old face of every path edge before the topology changes;
    // the faces of path edges at mesh vertices are removed here (the others by cutOneEdge or connectLoneContours),
    // and their edges show where to splice the path edges around the vertices
    HashMap<FaceId, std::array<EdgeId, 3>> vertFaces;
    for ( int contourId = 0; contourId < contours.size(); ++contourId )
    {
        const auto& inContour = contours[contourId].intersections;
        if ( inContour.size() < 2 )
            continue;
        auto& faces = res.oldEdgesInfo[contourId];
        faces.resize( inContour.size() - 1 );
        for ( int i = 0; i + 1 < inContour.size(); ++i )
        {
            const auto& a = inContour[i].primitiveId;
            const auto& b = inContour[i + 1].primitiveId;
            const auto va = std::get_if<VertId>( &a );
            const auto vb = std::get_if<VertId>( &b );
            if ( va && vb )
            {
                // the path edge is an edge of the old mesh
                const auto e = topology.findEdge( *va, *vb );
                assert( e );
                if ( !e )
                    continue;
                faces[i] = { topology.left( e ), topology.right( e ) };
            }
            else
                faces[i].left = faces[i].right = segmentFace( topology, inContour, i );
            if ( faces[i].left && ( va || vb ) )
            {
                auto [it, inserted] = vertFaces.try_emplace( faces[i].left );
                if ( inserted )
                    topology.getTriEdges( faces[i].left, it->second[0], it->second[1], it->second[2] );
            }
        }
    }
    for ( const auto& [f, edges] : vertFaces )
        topology.setLeft( edges[0], {} );

    // the contour and intersection at the other end of every path edge spliced at a mesh vertex, by the path edge from the vertex
    HashMap<EdgeId, std::pair<int, int>> spliceEnds;
    std::optional<CutTrianglePredicates> preds; // made on the first exact comparison, which Boolean contours never need
    // splices path edge e from mesh vertex v to intersection i of contour c in the place of old face f around v,
    // after the path edges already there that go before it counterclockwise: by the exact predicates if sortData is given
    auto spliceAtVert = [&] ( VertId v, FaceId f, EdgeId e, int c, int i )
    {
        const auto it = vertFaces.find( f );
        assert( it != vertFaces.end() );
        if ( it == vertFaces.end() )
            return;
        const auto& fe = it->second;
        const ThreeVertIds fv{ topology.org( fe[0] ), topology.org( fe[1] ), topology.org( fe[2] ) };
        const auto k = fv[0] == v ? 0 : fv[1] == v ? 1 : 2;
        assert( fv[k] == v );
        spliceEnds[e] = { c, i };
        const auto x0 = fe[k];
        if ( topology.next( x0 ) < oldEdgesSize ) // no path edges there yet
        {
            topology.splice( x0, e );
            return;
        }

        // point 0 is v, point 1 is the end of e, points 2... are the ends of the path edges there
        bool exact = false;
        if ( sortData )
        {
            if ( !preds )
                preds.emplace();
            preds->reset( preciseOwnTri( fv, mesh, *sortData ) );
            PreciseCutPoint corner;
            corner.seg[0] = preciseOwnVert( v, mesh, *sortData );
            preds->addPoint( VertId( 0 ), corner );
            preds->addPoint( VertId( 1 ), preciseCutPoint( mesh, contours, c, i, fe, fv, *sortData ) );
            int n = 2;
            for ( auto y = topology.next( x0 ); y >= oldEdgesSize; y = topology.next( y ) )
            {
                const auto [yc, yi] = spliceEnds.at( y );
                preds->addPoint( VertId( n++ ), preciseCutPoint( mesh, contours, yc, yi, fe, fv, *sortData ) );
            }
            exact = !preds->failed();
        }
        const auto& pv = mesh.points[v];
        const auto normal = cross( mesh.points[fv[( k + 1 ) % 3]] - pv, mesh.points[fv[( k + 2 ) % 3]] - pv );
        const auto p = contours[c].intersections[i].coordinate - pv;
        // the last path edge there that goes before e counterclockwise, or the old edge if none does
        auto findPlace = [&] ( bool byPreds )
        {
            auto x = x0;
            int n = 2;
            for ( auto y = topology.next( x ); y >= oldEdgesSize; y = topology.next( y ), ++n )
            {
                if ( byPreds )
                {
                    if ( !preds->ccw( VertId( 0 ), VertId( n ), VertId( 1 ) ) )
                        break;
                }
                else
                {
                    const auto [yc, yi] = spliceEnds.at( y );
                    if ( !( dot( cross( contours[yc].intersections[yi].coordinate - pv, p ), normal ) > 0 ) )
                        break;
                }
                x = y;
            }
            return x;
        };
        auto x = findPlace( exact );
        // a predicate can fail too, and then all its answers are discarded
        if ( exact && preds->failed() )
            x = findPlace( false );
        topology.splice( x, e );
    };

    for ( int contourId = 0; contourId < contours.size(); ++contourId )
    {
        const auto& inContour = contours[contourId].intersections;
        if ( inContour.size() < 2 )
            continue;
        const auto& faces = res.oldEdgesInfo[contourId];
        auto& path = res.paths[contourId];
        path.resize( inContour.size() - 1 );
        const bool closed = contours[contourId].closed;
        for ( int i = 0; i < inContour.size(); ++i )
        {
            const auto& inter = inContour[i];
            const auto vert = std::get_if<VertId>( &inter.primitiveId );
            // the path edges to and from this intersection
            const EdgeId in = i > 0 ? path[i - 1].sym() : EdgeId{};
            EdgeId out;
            if ( i + 1 < inContour.size() )
            {
                const auto nextVert = std::get_if<VertId>( &inContour[i + 1].primitiveId );
                const auto prevVert = i > 0 ? std::get_if<VertId>( &inContour[i - 1].primitiveId ) : nullptr;
                if ( vert && nextVert )
                    out = topology.findEdge( *vert, *nextVert );
                else if ( !vert && nextVert && prevVert && *prevVert == *nextVert )
                    out = in; // the contour touches this point from a mesh vertex and goes back along the same edge
                else if ( vert && closed && i + 2 == inContour.size() && inContour[1].primitiveId == inter.primitiveId )
                    out = path.front().sym(); // the same at the first point of a closed contour (out == in for x, v, x)
                else
                    out = topology.makeEdge();
                path[i] = out;
            }
            if ( vert )
            {
                // a path edge with an origin is already in place: an edge of the old mesh, or the edge back from a touched point
                if ( out && !topology.org( out ) )
                    spliceAtVert( *vert, faces[i].left, out, contourId, i + 1 );
                if ( in && !topology.org( in ) )
                    spliceAtVert( *vert, faces[i - 1].left, in, contourId, i - 1 );
                continue;
            }
            if ( closed && i + 1 == inContour.size() )
            {
                // the last intersection of a closed contour is the first one
                topology.splice( in, path.front() );
                continue;
            }
            const auto newVertId = topology.addVertId();
            mesh.points.autoResizeAt( newVertId ) = inter.coordinate;
            if ( out )
            {
                topology.setOrg( out, newVertId );
                if ( in )
                    topology.splice( in, out );
            }
            else
                topology.setOrg( in, newVertId );
            // add note to edgeData
            if ( auto e = std::get_if<EdgeId>( &inter.primitiveId ) )
            {
                auto& edgeData = res.edgeData[e->undirected()];
                edgeData.reserve( 5 ); // reserve a small amount to avoid reallocations in most common scenarios
                edgeData.emplace_back( EdgeIntersectionData{
                    .interOnEdge = IntersectionData{ContourId( contourId ),IntersectionId( i )},
                    .newVert = newVertId,
                    .orgEdgeInLeftTri = out,
                    .beforeSortIndex = int( edgeData.size() ) } );
            }
        }
    }
    return res;
}

void executeTriangulateContourPlan( Mesh& mesh, EdgeId e, HoleFillPlan& plan, FaceId oldFace, FaceMap* new2OldMap, NewEdgesMap* new2OldEdgeMap )
{
    const auto fsz0 = mesh.topology.faceSize();
    const auto uesz0 = mesh.topology.undirectedEdgeSize();
    executeHoleFillPlan( mesh, e, plan );
    if ( new2OldMap )
    {
        assert( oldFace.valid() );
        const auto fsz = mesh.topology.faceSize();
        new2OldMap->autoResizeSet( FaceId{ fsz0 }, fsz - fsz0, oldFace );
    }
    if ( new2OldEdgeMap )
    {
        assert( oldFace.valid() );
        for ( int ue = int( uesz0 ); ue < int( mesh.topology.undirectedEdgeSize() ); ++ue )
            new2OldEdgeMap->map[UndirectedEdgeId( ue )] = oldFace;
    }
}

void triangulateContour( Mesh& mesh, EdgeId e, FaceId oldFace, FaceMap* new2OldMap, NewEdgesMap* new2OldEdgeMap )
{
    auto plan = getPlanarHoleFillPlan( mesh, e );
    executeTriangulateContourPlan( mesh, e, plan, oldFace, new2OldMap, new2OldEdgeMap );
}

// a lone contour is inside one face crossing none of its edges (e.g. where a tip of the other mesh pokes through it),
// so it is an island in the hole of its face, or a slit if it is open
struct LoneFace
{
    FaceId f;
    std::array<EdgeId, 3> edges; // counterclockwise, before the cut
    ThreeVertIds verts; // verts[i] = org( edges[i] )
};

// finds the faces of lone contours, before the cut changes their topology
std::vector<LoneFace> findLoneFaces( const MeshTopology& topology, const OneMeshContours& contours )
{
    std::vector<FaceId> faces;
    for ( const auto& c : contours )
    {
        if ( c.intersections.size() > 1 && std::all_of( c.intersections.begin(), c.intersections.end(),
            [] ( const OneMeshIntersection& i ) { return std::holds_alternative<FaceId>( i.primitiveId ); } ) )
            faces.push_back( std::get<FaceId>( c.intersections.front().primitiveId ) );
    }
    std::sort( faces.begin(), faces.end() );
    faces.erase( std::unique( faces.begin(), faces.end() ), faces.end() );
    std::vector<LoneFace> res( faces.size() );
    for ( int i = 0; i < faces.size(); ++i )
    {
        auto& lf = res[i];
        lf.f = faces[i];
        topology.getTriEdges( lf.f, lf.edges[0], lf.edges[1], lf.edges[2] );
        for ( int k = 0; k < 3; ++k )
            lf.verts[k] = topology.org( lf.edges[k] );
    }
    return res;
}

// connects the islands of lone contours with the rest of the holes of their faces by the chords that split the holes in monotone parts,
// found on the exact predicates consistent with the boolean if sortData is given; the faces where that fails are added to badFaces
// and stay in the mesh unless another contour removed them
void connectLoneContours( Mesh& mesh, const OneMeshContours& contours, const std::vector<LoneFace>& loneFaces, const PreCutResult& preRes,
    UndirectedEdgeId firstNewEdge, const SortIntersectionsData* sortData, NewEdgesMap* new2OldEdgeMap, FaceBitSet& badFaces )
{
    if ( loneFaces.empty() )
        return;
    MR_TIMER;

    // the path edges inside every lone face, with the face on the left, and the contour points at their ends for the exact definitions
    std::vector<EdgePath> faceEdges( loneFaces.size() );
    HashMap<VertId, std::pair<int, int>> vertInters;
    for ( int i = 0; i < preRes.paths.size(); ++i )
    {
        for ( int j = 0; j < preRes.paths[i].size(); ++j )
        {
            const auto f = preRes.oldEdgesInfo[i][j].left;
            const auto it = std::lower_bound( loneFaces.begin(), loneFaces.end(), f, [] ( const LoneFace& lf, FaceId f ) { return lf.f < f; } );
            if ( it == loneFaces.end() || it->f != f )
                continue;
            auto& edges = faceEdges[it - loneFaces.begin()];
            const auto e = preRes.paths[i][j];
            edges.push_back( e );
            if ( e.undirected() >= firstNewEdge ) // a new edge is inside the face, while an old one has it on the left only
                edges.push_back( e.sym() );
            if ( !sortData )
                continue;
            vertInters[mesh.topology.org( e )] = { i, j };
            vertInters[mesh.topology.dest( e )] = { i, j + 1 };
        }
    }

    // the exact definition of a vertex on the loops of a lone face
    auto pointOf = [&] ( const LoneFace& lf, VertId v )
    {
        const auto it = vertInters.find( v );
        if ( it == vertInters.end() )
        {
            PreciseCutPoint res;
            res.seg[0] = preciseOwnVert( v, mesh, *sortData ); // a corner of the face
            return res;
        }
        return preciseCutPoint( mesh, contours, it->second.first, it->second.second, lf.edges, lf.verts, *sortData );
    };

    // the boundaries of the hole of every lone face
    std::vector<EdgeLoops> faceLoops( loneFaces.size() );
    EdgeBitSet inLoops( mesh.topology.edgeSize() );
    BitSet removedHere( loneFaces.size() ); // the faces that no other contour removed, to put back if their plans fail
    for ( int i = 0; i < loneFaces.size(); ++i )
    {
        const auto& lf = loneFaces[i];
        auto& edges = faceEdges[i];
        if ( mesh.topology.hasFace( lf.f ) ) // no other contour removed it
        {
            mesh.topology.setLeft( lf.edges[0], {} );
            edges.push_back( lf.edges[0] );
            removedHere.set( i );
        }
        for ( auto e : edges )
        {
            if ( mesh.topology.left( e ) || inLoops.test( e ) )
                continue;
            auto& loop = faceLoops[i].emplace_back();
            for ( auto le : leftRing( mesh.topology, e ) )
            {
                inLoops.set( le );
                loop.push_back( le );
            }
        }
    }

    // the faces are independent, so plan them in parallel
    std::vector<std::optional<HoleFillPlan>> plans( loneFaces.size() );
    ParallelFor( plans, [&] ( size_t i )
    {
        const auto& lf = loneFaces[i];
        const auto& loops = faceLoops[i];
        if ( sortData )
        {
            plans[i] = PlanarTriangulation::getMonotonePlan( mesh, loops, preciseOwnTri( lf.verts, mesh, *sortData ),
                [&] ( VertId v ) { return pointOf( lf, v ); } );
            return;
        }
        // the loops go counterclockwise around the normal of the face, of which only the dominant component and its sign matter;
        // the corners in doubles give a normal free of the rounding of the islands' points, which a sum over the loops would add;
        // normalized before the conversion to float, which could flush it to zero or overflow at extreme scales
        const auto& p = mesh.points;
        const auto normal = dirDblArea( Triangle3d{ Vector3d( p[lf.verts[0]] ), Vector3d( p[lf.verts[1]] ), Vector3d( p[lf.verts[2]] ) } );
        // corners on one line give no normal, so no plane to connect the islands in: the plan stays empty, and the face is reported
        if ( normal == Vector3d() )
            return;
        plans[i] = PlanarTriangulation::getMonotonePlan( mesh, loops, Vector3f( normal.normalized() ) );
    } );

    for ( int i = 0; i < loneFaces.size(); ++i )
    {
        auto& plan = plans[i];
        // every island or slit needs a chord to the rest of its face, so a plan without any leaves them unconnected
        // (as where the cut of a side has already filled the face again, ending an open contour on that side)
        if ( plan && !plan->items.empty() && isFillingMultipleEdgeFree( mesh.topology, *plan ) )
            executeTriangulateContourPlan( mesh, faceLoops[i].front().front(), *plan, loneFaces[i].f, nullptr, new2OldEdgeMap );
        else
        {
            badFaces.autoResizeSet( loneFaces[i].f );
            // the holes to fill are found by the cut paths, and the old sides of the face have none, so the face is put back;
            // no chord of it was added, so its old sides still make its ring
            if ( removedHere.test( i ) )
                mesh.topology.setLeft( loneFaces[i].edges[0], loneFaces[i].f );
        }
    }
}

[[maybe_unused]] void debugSortingInfo( EdgeId baseE,
                       const EdgeData& edgeData,
                       const std::vector<int>& res,
                       const std::vector<float>& dotProds,
                       const SortIntersectionsData* sortData )
{
    if ( edgeData.size() > 1 )
    {
        bool edgeInfoPrinted{false};
        for ( int i = 0; i + 1 < res.size(); ++i )
            //if ( dotProds[res[i]] == dotProds[res[i + 1]] )
        {
            if ( !edgeInfoPrinted )
            {
                spdlog::info( "Edge {}", (int)baseE );
                edgeInfoPrinted = true;
            }
            if ( sortData )
            {
                FaceId f1 = sortData->contours[edgeData[res[i]].interOnEdge.contourId][edgeData[res[i]].interOnEdge.intersectionId].tri();
                FaceId f2 = sortData->contours[edgeData[res[i + 1]].interOnEdge.contourId][edgeData[res[i + 1]].interOnEdge.intersectionId].tri();

                auto sharedEdge = sortData->otherMesh.topology.sharedEdge( f1, f2 );
                spdlog::info( "  {}", dotProds[res[i + 1]] - dotProds[res[i]] );
                spdlog::info( "   shared: ", (int)sharedEdge );
            }
        }
    }
}

enum class CompareResult
{
    Less,
    Unknown,
    Greater
};

/// compare intersections of two triangles with one edge along that edge;
/// returns Unknown only if both triangles share the same 3 vertices
CompareResult compareEdgeTrisIntersections( EdgeId intersectedEdge, const Mesh& mesh, const SortIntersectionsData& sortData,
    FaceId fl, FaceId fr )
{
    assert( fl != fr );

    const auto& otherTopology = sortData.otherMesh.topology;
    auto vsl = otherTopology.getTriVerts( fl );
    std::sort( vsl.begin(), vsl.end() );
    auto vsr = otherTopology.getTriVerts( fr );
    std::sort( vsr.begin(), vsr.end() );
    if ( vsl == vsr )
        return CompareResult::Unknown;

    std::array<PreciseVertCoords, 8> preciseVerts
    {
        preciseOwnVert( mesh.topology.org( intersectedEdge ), mesh, sortData ),
        preciseOwnVert( mesh.topology.dest( intersectedEdge ), mesh, sortData ),
        preciseOtherVert( vsl[0], sortData ),
        preciseOtherVert( vsl[1], sortData ),
        preciseOtherVert( vsl[2], sortData ),
        preciseOtherVert( vsr[0], sortData ),
        preciseOtherVert( vsr[1], sortData ),
        preciseOtherVert( vsr[2], sortData )
    };
    return segmentIntersectionOrder( preciseVerts ) ? CompareResult::Less : CompareResult::Greater;
}

/// compare intersections of two triangles with one edge along that edge;
/// if both triangles share the same 3 vertices then goes along the contours to other intersection in hope to resolve the ambiguity
CompareResult comparePropagateEdgeTrisIntersections( EdgeId intersectedEdge, const Mesh& mesh, const SortIntersectionsData& sortData,
     const EdgeIntersectionData& l, const EdgeIntersectionData& r )
{
    const auto & il = l.interOnEdge;
    const auto & ir = r.interOnEdge;
    const auto& lContour = sortData.contours[il.contourId];
    const auto& rContour = sortData.contours[ir.contourId];

    FaceId fl = lContour[il.intersectionId].tri();
    FaceId fr = rContour[ir.intersectionId].tri();
    const EdgeId el = lContour[il.intersectionId].edge;
    const EdgeId er = rContour[ir.intersectionId].edge;
    assert( el.undirected() == intersectedEdge.undirected() );
    assert( er.undirected() == intersectedEdge.undirected() );
    auto res = compareEdgeTrisIntersections( intersectedEdge, mesh, sortData, fl, fr );
    if ( res != CompareResult::Unknown )
        return res;

    const auto& tp = mesh.topology;

    bool edgeATriB = lContour[il.intersectionId].isEdgeATriB();
    bool sameContour = il.contourId == ir.contourId;
    int stepRight = el == er ? 1 : -1;

    // finds next/prev intersection on edge in the contour
    auto getNextPrev = [&] ( IntersectionId interData, IntersectionId stopInter, bool left, bool next )->IntersectionId
    {
        const auto& contour = left ? lContour : rContour;
        bool closed = isClosed( contour );
        int step = left ? 1 : stepRight;
        if ( !next )
            step *= -1;
        IntersectionId nextL = interData;
        int size = int( contour.size() );
        for ( ;;)
        {
            int nextIndex = nextL + step;
            if ( !closed && ( nextIndex < 0 || nextIndex >= size ) )
                return {}; // reached end of non closed contour
            nextL = IntersectionId( ( nextIndex + size ) % size );
            if ( closed && nextL + 1 == size )
                continue;
            if ( nextL == stopInter )
                return {}; // reached stop intersection in the contour
            if ( contour[nextL].isEdgeATriB() == edgeATriB )
                return nextL; // return next/prev intersection (on edge)
        }
    };

    bool tryNext = true;
    bool tryPrev = true;

    IntersectionId lNext = il.intersectionId;
    IntersectionId rNext = ir.intersectionId;
    IntersectionId lPrev = il.intersectionId;
    IntersectionId rPrev = ir.intersectionId;
    EdgeId lastCommonEdgeNext = intersectedEdge;
    EdgeId lastCommonEdgePrev = intersectedEdge;
    // check if next/prev intersection can determine sort
    auto checkOther = [&] ( bool next )->CompareResult
    {
        auto& tryThis = next ? tryNext : tryPrev;
        assert( tryThis );

        const auto startL = next ? lNext : lPrev;
        const auto startR = next ? rNext : rPrev;

        auto& lOtherRef = next ? lNext : lPrev;
        auto& rOtherRef = next ? rNext : rPrev;
        auto& lastCommonEdgeRef = next ? lastCommonEdgeNext : lastCommonEdgePrev;
        auto otherL = getNextPrev( lOtherRef, sameContour ? rOtherRef : lOtherRef, true, next );
        if ( !otherL )
        {
            tryThis = false; // terminal (end of contour reached)
            return CompareResult::Unknown;
        }
        auto otherR = getNextPrev( rOtherRef, sameContour ? lOtherRef : rOtherRef, false, next );
        if ( !otherR )
        {
            tryThis = false; // terminal (end of contour reached)
            return CompareResult::Unknown;
        }
        lOtherRef = otherL;
        rOtherRef = otherR;
        auto otherEL = lContour[lOtherRef].edge.undirected();
        auto otherER = rContour[rOtherRef].edge.undirected();
        bool lReturned = otherEL == lastCommonEdgeRef.undirected();
        bool rReturned = otherER == lastCommonEdgeRef.undirected();
        if ( lReturned || rReturned )
        {
            // if one of candidates return - terminal, but still can be determined
            tryThis = false; // terminal
            // if both of candidates return to base edge sort cannot be determined
            if ( lReturned && rReturned )
                return CompareResult::Unknown;

            FaceId fl;
            FaceId fr;
            if ( lReturned )
            {
                fl = lContour[lOtherRef].tri();
                fr = rContour[startR].tri();
            }
            else
            {
                assert( rReturned );
                fl = lContour[startL].tri();
                fr = rContour[rOtherRef].tri();
            }
            return compareEdgeTrisIntersections( lastCommonEdgeRef, mesh, sortData, fl, fr );
        }

        if ( otherEL != otherER )
        {
            // following assert is valid for common two objects boolean case, while for self-boolean it might be violated
            // keeping it for better understanding whats going on here, also might be useful for debugging two objects boolean failures

            //assert(
            //    ( otherEL == tp.next( lastCommonEdgeRef ).undirected() && otherER == tp.prev( lastCommonEdgeRef.sym() ).undirected() ) ||
            //    ( otherER == tp.next( lastCommonEdgeRef ).undirected() && otherEL == tp.prev( lastCommonEdgeRef.sym() ).undirected() ) ||
            //    ( otherEL == tp.prev( lastCommonEdgeRef ).undirected() && otherER == tp.next( lastCommonEdgeRef.sym() ).undirected() ) ||
            //    ( otherER == tp.prev( lastCommonEdgeRef ).undirected() && otherEL == tp.next( lastCommonEdgeRef.sym() ).undirected() ) );

            // determined condition, intersections leave face in different edges (not returned)
            if ( otherEL == tp.next( lastCommonEdgeRef ).undirected() || otherEL == tp.prev( lastCommonEdgeRef ).undirected() )
                return sortData.isOtherA ? CompareResult::Less : CompareResult::Greater; // terminal
            else if ( otherER == tp.next( lastCommonEdgeRef ).undirected() || otherER == tp.prev( lastCommonEdgeRef ).undirected() )
                return sortData.isOtherA ? CompareResult::Greater : CompareResult::Less; // terminal
            else
            {
                // TODO: support this case
                // we can be here only if doing self-boolean of non-closed contour passing through vertex
                tryThis = false; // for now just terminate, for simplicity
                return CompareResult::Unknown;
            }
        }

        // undetermined condition, but not terminal (intersections leave face in same edge (not returned))
        assert( otherEL == otherER && !lReturned && !rReturned );
        if ( otherEL == tp.next( lastCommonEdgeRef ).undirected() )
            lastCommonEdgeRef = tp.next( lastCommonEdgeRef );
        else if ( otherEL == tp.prev( lastCommonEdgeRef ).undirected() )
            lastCommonEdgeRef = tp.prev( lastCommonEdgeRef );
        else if ( otherEL == tp.prev( lastCommonEdgeRef.sym() ).undirected() )
            lastCommonEdgeRef = tp.prev( lastCommonEdgeRef.sym() ).sym();
        else
            lastCommonEdgeRef = tp.next( lastCommonEdgeRef.sym() ).sym();

        FaceId fl = lContour[lOtherRef].tri();
        FaceId fr = rContour[rOtherRef].tri();

        if ( fl == fr )
            return CompareResult::Unknown; // go next if we came to same intersection 

        return compareEdgeTrisIntersections( lastCommonEdgeRef, mesh, sortData, fl, fr );

    };
    bool lPassedFullRing = false;
    bool rPassedFullRing = false;
    res = CompareResult::Unknown;
    for ( ; tryNext || tryPrev; )
    {
        if ( tryNext )
            res = checkOther( true );
        if ( res != CompareResult::Unknown )
            return res;
        if ( tryPrev )
            res = checkOther( false );
        if ( res != CompareResult::Unknown )
            return res;

        if ( !lPassedFullRing && ( lNext == il.intersectionId || lPrev == il.intersectionId ) )
            lPassedFullRing = true;
        if ( !rPassedFullRing && ( rNext == ir.intersectionId || rPrev == ir.intersectionId ) )
            rPassedFullRing = true;

        if ( lPassedFullRing && rPassedFullRing )
            return CompareResult::Unknown; // both contours passed a round, so break infinite loop
    }

    return res;
}

void sortEdgeInfo( const Mesh& mesh, const OneMeshContours& contours, EdgeData& edgeData,
    const SortIntersectionsData* sortData ) // it will probably be useful for precise sorting
{
    assert( !edgeData.empty() );
    const auto& intInfo = edgeData.front().interOnEdge;
    EdgeId baseEdge = std::get<EdgeId>( contours[intInfo.contourId].intersections[intInfo.intersectionId].primitiveId );

    if ( sortData )
    {
        auto pred = [&]( const EdgeIntersectionData& l, const EdgeIntersectionData& r ) -> bool
        {
            auto res = comparePropagateEdgeTrisIntersections( baseEdge, mesh, *sortData, l, r );
            if ( res != CompareResult::Unknown )
                return res == CompareResult::Less;

            assert( mesh.points[l.newVert] == mesh.points[r.newVert] );
            assert( !"cannot compare" );
            return false;
        };
        std::sort( edgeData.begin(), edgeData.end(), pred );
    }
    else
    {
        // sort using floating-point coordinates of new vertices
        std::vector<double> dotProds( edgeData.size() );
        Vector3d orgPoint{ mesh.orgPnt( baseEdge ) };
        auto abVec = Vector3d{ mesh.destPnt( baseEdge ) } - orgPoint;
        for ( int i = 0; i < edgeData.size(); ++i )
            dotProds[i] = dot( Vector3d{ mesh.points[edgeData[i].newVert] } - orgPoint, abVec );

        auto pred = [&]( const EdgeIntersectionData& l, const EdgeIntersectionData& r ) -> bool
        {
            return dotProds[l.beforeSortIndex] < dotProds[r.beforeSortIndex];
        };
        std::sort( edgeData.begin(), edgeData.end(), pred );
    }

    // DEBUG Output
    //debugSortingInfo( baseE, edgeData, res, dotProds, sortData );
}

//            ^
//            | top edge
//            |
// <---------- ----------->
// left edge  ^  right edge
//            | bot edge
//            |
// left and right are already connected
void connectEdges( MeshTopology& topology, EdgeId botEdge, EdgeId topEdge, EdgeId leftEdge, EdgeId rightEdge )
{
    assert( botEdge.valid() && topEdge.valid() && ( leftEdge.valid() || rightEdge.valid() ) );

    if ( !leftEdge.valid() )
    {
        topology.splice( rightEdge, topEdge );
        topology.splice( topEdge, botEdge.sym() );
    }
    else if ( !rightEdge.valid() )
    {
        topology.splice( topEdge, leftEdge );
        topology.splice( leftEdge, botEdge.sym() );
    }
    else
    {
        topology.splice( rightEdge, topEdge );
        topology.splice( leftEdge, botEdge.sym() );
    }
}

// cuts one edge and connects all intersecting contours with pieces
void cutOneEdge( Mesh& mesh,
                 const EdgeData& edgeData, const OneMeshContours& contours,
                 FaceMap* new2OldMap, NewEdgesMap* new2OldEdgeMap )
{
    assert( !edgeData.empty() );

    const auto& intInfo = std::find_if( edgeData.begin(), edgeData.end(), [] ( const auto& data )
    {
        return data.beforeSortIndex == 0;
    } )->interOnEdge;
    EdgeId baseEdge = std::get<EdgeId>( contours[intInfo.contourId].intersections[intInfo.intersectionId].primitiveId );

    // will need this to restore lost face on first or last intersection (only for open contours)
    FaceId oldLeft = mesh.topology.left( baseEdge );
    FaceId oldRight = mesh.topology.left( baseEdge.sym() );

    // remove incident faces
    mesh.topology.setLeft( baseEdge, FaceId{} );
    mesh.topology.setLeft( baseEdge.sym(), FaceId{} );

    EdgeId e = baseEdge;
    // disconnect edge e from its origin
    EdgeId e0;
    {
        EdgeId ePrev = mesh.topology.prev( e );
        if ( ePrev != e )
            mesh.topology.splice( ePrev, e );
        // e now becomes the second part of split edge, add first part to it
        e0 = mesh.topology.makeEdge();
        if ( new2OldEdgeMap )
        {
            new2OldEdgeMap->splitEdges.autoResizeSet( e0.undirected() );
            new2OldEdgeMap->map[e0.undirected()] = baseEdge;
        }
        if ( ePrev != e )
            mesh.topology.splice( ePrev, e0 );
    }

    bool isAllLeftOnly = true;
    bool isAllRightOnly = true;
    for ( int i = 0; i < edgeData.size(); ++i )
    {
        const auto& vertEdge = edgeData[i].orgEdgeInLeftTri;
        const auto& interIndex = edgeData[i].interOnEdge;
        const auto& inter = contours[interIndex.contourId].intersections[interIndex.intersectionId];

        bool isBaseSym = std::get<EdgeId>( inter.primitiveId ).sym() == baseEdge;

        EdgeId leftEdge, rightEdge;
        EdgeId& baseleft = isBaseSym ? rightEdge : leftEdge;
        EdgeId& baserigth = isBaseSym ? leftEdge : rightEdge;

        baseleft = vertEdge;
        baserigth = baseleft.valid() ? mesh.topology.next( baseleft ) : mesh.topology.edgePerVertex()[edgeData[i].newVert];
        if ( baseleft == baserigth )
            baserigth = EdgeId{};

        EdgeId lastEdge = e;
        if ( i + 1 < edgeData.size() )
        {
            lastEdge = mesh.topology.makeEdge();
            if ( new2OldEdgeMap )
            {
                new2OldEdgeMap->map[lastEdge.undirected()] = isBaseSym ? baseEdge.sym() : baseEdge;
                new2OldEdgeMap->splitEdges.autoResizeSet( lastEdge.undirected() );
            }
        }

        if ( isAllLeftOnly && rightEdge.valid() )
            isAllLeftOnly = false;

        if ( isAllRightOnly && leftEdge.valid() )
            isAllRightOnly = false;

        connectEdges( mesh.topology, e0, lastEdge, leftEdge, rightEdge );

        e0 = lastEdge;
    }

    // fix triangle if this was last or first
    if ( isAllLeftOnly && oldRight.valid() )
        triangulateContour( mesh, e0.sym(), oldRight, new2OldMap, new2OldEdgeMap );
    if ( isAllRightOnly && oldLeft.valid() )
        triangulateContour( mesh, e0, oldLeft, new2OldMap, new2OldEdgeMap );
}

// this function cut mesh edge and connects it with result path,
// after it each path edge left and right faces are invalid (they are removed)
void cutEdgesIntoPieces( Mesh& mesh,
                         EdgeDataMap&& edgeData, const OneMeshContours& contours,
                         const SortIntersectionsData* sortData,
                         FaceMap* new2OldMap, NewEdgesMap* new2OldEdgeMap )
{
    MR_TIMER;
    // sort each edge intersections in parallel
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, edgeData.subcnt(), 1 ),
        [&] ( const tbb::blocked_range<size_t>& range )
    {
        assert( range.begin() + 1 == range.end() );
        for ( size_t i = range.begin(); i != range.end(); ++i )
        {
            edgeData.with_submap( i, [&] ( const EdgeDataMap::EmbeddedSet& subSet )
            {
                // const_cast here is safe, we don't write to map, just sort internal data
                for ( auto& edgeInfo : const_cast< EdgeDataMap::EmbeddedSet& >( subSet ) )
                {
                    sortEdgeInfo( mesh, contours, edgeInfo.second, sortData );
                }
            } );
        }
    } );
    // cut all
    for ( const auto& edgeInfo : edgeData )
        cutOneEdge( mesh, edgeInfo.second, contours, new2OldMap, new2OldEdgeMap );
}

void prepareFacesMap( const MeshTopology& topology, FaceMap& new2OldMap )
{
    MR_TIMER;
    new2OldMap.resize( topology.lastValidFace() + 1 );
    for ( auto f : topology.getValidFaces() )
        new2OldMap[f] = f;
}

// Checks if cut mesh has valid loops in intersections
// if error occurs returns bad faces bit set, otherwise returns empty one
FaceBitSet getBadFacesAfterCut( const MeshTopology& topology, const PreCutResult& preRes )
{
    MR_TIMER;
    FaceBitSet badFacesBS( topology.getValidFaces().size() );
    EdgeBitSet visited( topology.edgeSize() );
    // whether the path from e on through the points of two edges ends at a point of one edge
    auto dangles = [&] ( EdgeId e )
    {
        for ( auto d = e.sym();; )
        {
            const auto n = topology.next( d );
            if ( n == d )
                return true;
            if ( n == e || topology.next( n ) != d )
                return false;
            d = n.sym();
        }
    };
    for ( int pathId = 0; pathId < preRes.paths.size(); ++pathId )
    {
        const auto& path = preRes.paths[pathId];
        for ( int edgeId = 0; edgeId < path.size(); ++edgeId )
        {
            const auto e0 = path[edgeId];
            const auto oldf = preRes.oldEdgesInfo[pathId][edgeId].left;
            if ( !oldf )
                continue;
            if ( visited.test( e0 ) || visited.test( e0.sym() ) )
                continue;
            // the hole meets the sym of a path edge by itself where the path goes along the edge and back (to a touched point
            // or along an edge of the mesh, also around the start of the path: then it ends where it starts, closed or not),
            // and where the edge dangles in the hole (at the end of an open contour inside a face, or in a slit)
            const auto prev = edgeId > 0 ? path[edgeId - 1] : path.back();
            const auto next = edgeId + 1 < path.size() ? path[edgeId + 1] : path.front();
            if ( prev == e0.sym() || next == e0.sym() )
                continue;
            for ( auto e : leftRing( topology, e0 ) )
            {
                visited.set( e );
                if ( e == e0.sym() && !dangles( e0 ) && !dangles( e0.sym() ) )
                    badFacesBS.autoResizeSet( oldf );
            }
        }
    }
    return badFacesBS;
}

// a run of consecutive segments of one contour inside one face
struct ContourPiece
{
    FaceId f;
    int c = -1;
    int first = -1; // the first segment
    int num = 0; // the number of segments, which wrap past the end of a closed contour
};

// the pieces of all contours, sorted by faces
std::vector<ContourPiece> findContourPieces( const MeshTopology& topology, const OneMeshContours& contours )
{
    MR_TIMER;
    // the faces of the segments of all contours in one array, contour c from firstSeg[c]
    std::vector<int> firstSeg( contours.size() + 1, 0 );
    for ( int c = 0; c < contours.size(); ++c )
        firstSeg[c + 1] = firstSeg[c] + std::max( int( contours[c].intersections.size() ) - 1, 0 );
    std::vector<FaceId> segFaces( firstSeg.back() );
    // in parallel over all segments, since one contour can have most of them, and they read the topology at random;
    // over ranges, not by ParallelFor, so that each range finds its first contour once
    tbb::parallel_for( tbb::blocked_range<int>( 0, firstSeg.back() ), [&] ( const tbb::blocked_range<int>& range )
    {
        int c = int( std::upper_bound( firstSeg.begin(), firstSeg.end(), range.begin() ) - firstSeg.begin() ) - 1;
        for ( int k = range.begin(); k < range.end(); ++k )
        {
            while ( k >= firstSeg[c + 1] )
                ++c;
            segFaces[k] = segmentFace( topology, contours[c].intersections, k - firstSeg[c] );
        }
    } );

    tbb::enumerable_thread_specific<std::vector<ContourPiece>> threadPieces;
    ParallelFor( contours, threadPieces, [&] ( size_t c, std::vector<ContourPiece>& pieces )
    {
        const int numSegs = firstSeg[c + 1] - firstSeg[c];
        const auto* faces = segFaces.data() + firstSeg[c];
        // the runs of segments in one face; a closed contour starts them after a change of the face, so that no run is split at the end
        int start = 0;
        if ( contours[c].closed )
        {
            while ( start < numSegs && faces[start] == faces[( start + numSegs - 1 ) % numSegs] )
                ++start;
            if ( start == numSegs )
                start = 0; // the whole contour is in one face
        }
        for ( int k = 0; k < numSegs; )
        {
            const int first = ( start + k ) % numSegs;
            const auto f = faces[first];
            int num = 1;
            while ( k + num < numSegs && faces[( first + num ) % numSegs] == f )
                ++num;
            k += num;
            if ( f ) // not along an edge of the mesh
                pieces.push_back( { f, int( c ), first, num } );
        }
    } );
    std::vector<ContourPiece> res;
    size_t total = 0;
    for ( const auto& pieces : threadPieces )
        total += pieces.size();
    res.reserve( total );
    for ( const auto& pieces : threadPieces )
        res.insert( res.end(), pieces.begin(), pieces.end() );
    tbb::parallel_sort( res.begin(), res.end(), [] ( const ContourPiece& a, const ContourPiece& b )
    {
        return std::tie( a.f, a.c, a.first ) < std::tie( b.f, b.c, b.first );
    } );
    return res;
}

// a segment of a contour in a face, projected on the plane of the two axes least aligned with the normal of the face
struct FaceSegment
{
    int c = -1;
    int i = -1; // from intersection i to i + 1
    int p = -1; // the index of intersection i among the points of the face, and p + 1 of intersection i + 1
    Vector2d a, b;
    Box2d box;
    bool loose = false; // an end may be far from its coordinate, and the box is around its exact point then
};

// 1 if point c is on the left of line ab for sure, -1 if on the right, 0 if the errors of the coordinates, below tol along each axis, may change it
int approxSide( const Vector2d& a, const Vector2d& b, const Vector2d& c, double tol )
{
    const double ux = b.x - a.x, uy = b.y - a.y, wx = c.x - a.x, wy = c.y - a.y;
    const double l = ux * wy, r = uy * wx;
    const double area = l - r;
    // the differences move by 2 tol, and the products and their difference add rounding errors below 2^-52 of the sum of the products
    const double err = 2 * tol * ( std::abs( ux ) + std::abs( uy ) + std::abs( wx ) + std::abs( wy ) ) + 8 * tol * tol
        + 0x1p-50 * ( std::abs( l ) + std::abs( r ) );
    return int( area > err ) - int( area < -err );
}

// 1 if two segments cross for sure, -1 if they do not, 0 if the errors of the coordinates, below tol along each axis, may change the answer;
// for tol = 0, the coordinates decide, and 0 means that an end of a segment is on the line of the other one, or within rounding of it
int approxCross( const FaceSegment& s, const FaceSegment& t, double tol )
{
    const int sc = approxSide( s.a, s.b, t.a, tol ), sd = approxSide( s.a, s.b, t.b, tol );
    if ( sc != 0 && sc == sd )
        return -1;
    const int ta = approxSide( t.a, t.b, s.a, tol ), tb = approxSide( t.a, t.b, s.b, tol );
    if ( ta != 0 && ta == tb )
        return -1;
    return sc != 0 && sd != 0 && ta != 0 && tb != 0 ? 1 : 0;
}

// the largest absolute value of the coordinates of a point
float maxAbsCoord( const Vector3f& p )
{
    return std::max( { std::abs( p.x ), std::abs( p.y ), std::abs( p.z ) } );
}

// finds whether two segments of the contours cross inside a face, one face at a time, keeping its buffers for the next face: the coordinates
// decide the pairs that their conservative error bounds allow, the exact predicates consistent with the boolean decide the others (with sortData)
class FaceCrossingsFinder
{
public:
    FaceCrossingsFinder( const Mesh& mesh, const OneMeshContours& contours, const SortIntersectionsData* sortData )
        : mesh_( mesh ), contours_( contours ), sortData_( sortData )
    {
        // the points of mesh B move to mesh A by rigidB2A, and the coordinates of the contours of mesh B back by its inverse
        if ( sortData && sortData->rigidB2A )
        {
            ( sortData->isOtherA ? own2A_ : other2own_ ) = *sortData->rigidB2A;
            if ( sortData->isOtherA )
                other2own_ = sortData->rigidB2A->inverse();
        }
    }

    // whether two segments of the pieces of one face cross
    bool find( std::span<const ContourPiece> pieces )
    {
        setFace_( pieces.front().f );
        makeSegments_( pieces );
        predsReady_ = false;
        return sweep_();
    }

private:
    // sets the face, the bound of the errors of the coordinates in it, the projection, and the exact corners
    void setFace_( FaceId f )
    {
        mesh_.topology.getTriEdges( f, edges_[0], edges_[1], edges_[2] );
        verts_ = { mesh_.topology.org( edges_[0] ), mesh_.topology.org( edges_[1] ), mesh_.topology.org( edges_[2] ) };
        for ( int k = 0; k < 3; ++k )
            corners_[k] = mesh_.points[verts_[k]];

        tol_ = 0;
        if ( sortData_ )
        {
            float maxAbs = 0;
            for ( const auto& p : corners_ )
                maxAbs = std::max( { maxAbs, maxAbsCoord( p ), maxAbsCoord( own2A_( p ) ) } );
            tol_ = tolOf_( maxAbs );
            for ( int k = 0; k < 3; ++k )
                ownInt_[k] = preciseOwnVert( verts_[k], mesh_, *sortData_ ).pt;
        }

        // drop the axis most aligned with the normal
        const auto n = dirDblArea( Vector3d( corners_[0] ), Vector3d( corners_[1] ), Vector3d( corners_[2] ) );
        const int drop = std::abs( n.x ) >= std::abs( n.y ) && std::abs( n.x ) >= std::abs( n.z ) ? 0 : std::abs( n.y ) >= std::abs( n.z ) ? 1 : 2;
        kx_ = drop == 0 ? 1 : 0;
        ky_ = drop == 2 ? 1 : 2;
    }

    // the bound of the errors along each axis of the coordinates of the intersections and of the vertices moved into this mesh
    // from their exact points, near points whose coordinates in both meshes are below m = maxAbs by absolute value; with u = 2^-24:
    // the integer grid adds half a unit (below one after a rotation), the rounding of the intersections to float u m, rigidB2A in float
    // on a vertex of mesh B below 11 u m, and its inverse in float (the entries of its matrix are off by up to 10 u) below 37 u m < 2^-18 m
    double tolOf_( float maxAbs ) const
    {
        const double unit = 1 / sortData_->converter.invRange;
        return 2 * unit + 0x1p-18 * ( maxAbs + unit );
    }

    Vector2d project_( const Vector3f& p ) const
    {
        return { p[kx_], p[ky_] };
    }

    // a box of the exact intersection i of contour c if it is a loose point, otherwise nothing: findTriangleSegmentIntersectionPrecise
    // takes the middle of the part of the segment inside the triangle if both ends of the segment are exactly in the plane of the triangle,
    // which can be far from the exact point; the exact point is on the segment and inside the triangle, so in the boxes of the coordinates
    // of their vertices moved into this mesh as the contours are, each expanded by the bound of the errors of its vertices
    std::optional<Box2d> looseBox_( int c, int i ) const
    {
        if ( !sortData_ )
            return {};
        const auto& prim = contours_[c].intersections[i].primitiveId;
        const auto& et = sortData_->contours[c][i];
        const auto& other = sortData_->otherMesh;
        // the intersection of the given box of this mesh with the box of the vertices vs of the other mesh
        auto withOther = [&] ( const Box2d& own, std::span<const VertId> vs )
        {
            Box2d res;
            float maxAbs = 0;
            for ( auto v : vs )
            {
                const auto q = other2own_( other.points[v] );
                maxAbs = std::max( { maxAbs, maxAbsCoord( other.points[v] ), maxAbsCoord( q ) } );
                res.include( project_( q ) );
            }
            res = own.intersection( res.expanded( Vector2d::diagonal( tolOf_( maxAbs ) ) ) );
            assert( res.valid() ); // both boxes contain the exact point
            return res;
        };
        // the box of num corners of the face from corner k
        auto ownBox = [&] ( int k, int num )
        {
            Box2d res;
            for ( int j = 0; j < num; ++j )
                res.include( project_( corners_[( k + j ) % 3] ) );
            return res.expanded( Vector2d::diagonal( tol_ ) );
        };
        if ( const auto pe = std::get_if<EdgeId>( &prim ) )
        {
            // side k of this face crossing a triangle of the other mesh
            int k = 0;
            while ( k < 3 && edges_[k].undirected() != pe->undirected() )
                ++k;
            if ( k == 3 )
                return ownBox( 0, 3 ); // not a side, which breaks a precondition
            const auto t = other.topology.getTriVerts( et.tri() );
            const auto a = preciseOtherVert( t[0], *sortData_ ).pt, b = preciseOtherVert( t[1], *sortData_ ).pt;
            const auto d = preciseOtherVert( t[2], *sortData_ ).pt;
            for ( int j = 0; j < 2; ++j )
                if ( detail::orient3dVolume( a, b, d, ownInt_[( k + j ) % 3] ) != 0 )
                    return {};
            return withOther( ownBox( k, 2 ), t );
        }
        if ( std::holds_alternative<FaceId>( prim ) )
        {
            // an edge of the other mesh crossing this face
            const std::array<VertId, 2> e{ other.topology.org( et.edge ), other.topology.dest( et.edge ) };
            for ( auto v : e )
                if ( detail::orient3dVolume( ownInt_[0], ownInt_[1], ownInt_[2], preciseOtherVert( v, *sortData_ ).pt ) != 0 )
                    return {};
            return withOther( ownBox( 0, 3 ), e );
        }
        return {}; // a corner
    }

    // makes the projected segments of the pieces, numbering the points of the face piece by piece
    void makeSegments_( std::span<const ContourPiece> pieces )
    {
        segs_.clear();
        int numPoints = 0;
        for ( const auto& p : pieces )
        {
            const auto& inContour = contours_[p.c].intersections;
            const int numSegs = int( inContour.size() ) - 1;
            auto looseA = looseBox_( p.c, p.first );
            for ( int k = 0; k < p.num; ++k )
            {
                auto& s = segs_.emplace_back();
                s.c = p.c;
                s.i = ( p.first + k ) % numSegs;
                s.p = numPoints + k;
                s.a = project_( inContour[s.i].coordinate );
                s.b = project_( inContour[s.i + 1].coordinate );
                const auto looseB = looseBox_( p.c, s.i + 1 );
                s.box = looseA.value_or( Box2d( s.a, s.a ) );
                s.box.include( looseB.value_or( Box2d( s.b, s.b ) ) );
                s.loose = looseA || looseB;
                looseA = looseB;
            }
            numPoints += p.num + 1;
        }
    }

    // whether two segments touch at their common end, so that they do not cross: they follow each other along a contour or meet at a mesh vertex
    bool touch_( const FaceSegment& s, const FaceSegment& t ) const
    {
        if ( s.c == t.c )
        {
            const int numSegs = int( contours_[s.c].intersections.size() ) - 1;
            const int d = std::abs( s.i - t.i );
            if ( d == 1 || ( contours_[s.c].closed && d == numSegs - 1 ) )
                return true;
        }
        for ( int sk = 0; sk < 2; ++sk )
            if ( const auto sv = std::get_if<VertId>( &contours_[s.c].intersections[s.i + sk].primitiveId ) )
                for ( int tk = 0; tk < 2; ++tk )
                    if ( const auto tv = std::get_if<VertId>( &contours_[t.c].intersections[t.i + tk].primitiveId ); tv && *tv == *sv )
                        return true;
        return false;
    }

    // whether two segments cross by the exact predicates, which get all points of the face on the first pair that the coordinates leave open,
    // so that the shifts of the points defined alike (see CutTrianglePredicates), and so the answers, do not depend on the order of the pairs;
    // a failure (points that break a precondition, or that the predicates cannot tell apart) makes the face bad anyway
    bool exactCross_( const FaceSegment& s, const FaceSegment& t )
    {
        if ( !predsReady_ )
        {
            preds_.reset( preciseOwnTri( verts_, mesh_, *sortData_ ) );
            for ( size_t k = 0; k < segs_.size(); ++k )
            {
                const auto& x = segs_[k];
                preds_.addPoint( VertId( x.p ), preciseCutPoint( mesh_, contours_, x.c, x.i, edges_, verts_, *sortData_ ) );
                if ( k + 1 == segs_.size() || segs_[k + 1].p != x.p + 1 ) // the last segment of a piece
                    preds_.addPoint( VertId( x.p + 1 ), preciseCutPoint( mesh_, contours_, x.c, x.i + 1, edges_, verts_, *sortData_ ) );
            }
            predsReady_ = true;
        }
        if ( preds_.failed() )
            return true;
        const VertId a( s.p ), b( s.p + 1 ), c( t.p ), d( t.p + 1 );
        return ( preds_.ccw( a, b, c ) != preds_.ccw( a, b, d ) && preds_.ccw( c, d, a ) != preds_.ccw( c, d, b ) ) || preds_.failed();
    }

    // whether two segments cross, by a sweep along the first axis over their boxes, till the first crossing
    bool sweep_()
    {
        order_.resize( segs_.size() );
        std::iota( order_.begin(), order_.end(), 0 );
        std::sort( order_.begin(), order_.end(), [&] ( int x, int y ) { return segs_[x].box.min.x < segs_[y].box.min.x; } );
        active_.clear();
        for ( int j : order_ )
        {
            const auto& t = segs_[j];
            std::erase_if( active_, [&] ( int i ) { return segs_[i].box.max.x + 2 * tol_ < t.box.min.x; } );
            const auto tBox = t.box.expanded( Vector2d::diagonal( 2 * tol_ ) );
            for ( int i : active_ )
            {
                const auto& s = segs_[i];
                if ( !s.box.intersects( tBox ) || touch_( s, t ) )
                    continue;
                // without sortData, tol is zero, and 0 means that an end of a segment is on the line of the other one: touching, no crossing;
                // the coordinates of a loose point tell nothing
                const int r = s.loose || t.loose ? 0 : approxCross( s, t, tol_ );
                if ( r > 0 || ( r == 0 && sortData_ && exactCross_( s, t ) ) )
                    return true;
            }
            active_.push_back( j );
        }
        return false;
    }

    const Mesh& mesh_;
    const OneMeshContours& contours_;
    const SortIntersectionsData* sortData_ = nullptr;
    AffineXf3f own2A_, other2own_; // move the points of this mesh to mesh A, and those of the other mesh into the coordinates of the contours
    std::array<EdgeId, 3> edges_; // counterclockwise
    ThreeVertIds verts_; // verts_[k] = org( edges_[k] )
    std::array<Vector3f, 3> corners_;
    double tol_ = 0; // the bound of the errors of the coordinates of the intersections along each axis
    int kx_ = 0, ky_ = 1; // the projection axes
    std::array<Vector3i, 3> ownInt_; // the exact corners, if sortData is given
    std::vector<FaceSegment> segs_;
    std::vector<int> order_, active_; // the buffers of the sweep
    CutTrianglePredicates preds_;
    bool predsReady_ = false; // whether preds_ have the points of the face
};

// finds the faces of the mesh where two segments of the contours cross, before the cut; the segments of one face are checked
// only if it has two pieces of the contours or a piece of three segments; two segments that follow each other along a contour
// or have a common mesh vertex do not cross
FaceBitSet findCrossingFaces( const Mesh& mesh, const OneMeshContours& contours, const SortIntersectionsData* sortData )
{
    MR_TIMER;
    const auto pieces = findContourPieces( mesh.topology, contours );
    // the pieces of the faces with two pieces or with a piece of three segments, where two segments may cross
    std::vector<std::span<const ContourPiece>> faces;
    for ( int i = 0; i < pieces.size(); )
    {
        int j = i + 1;
        while ( j < pieces.size() && pieces[j].f == pieces[i].f )
            ++j;
        if ( j > i + 1 || pieces[i].num >= 3 )
            faces.emplace_back( pieces.data() + i, j - i );
        i = j;
    }

    tbb::enumerable_thread_specific<FaceCrossingsFinder> finders( [&] { return FaceCrossingsFinder( mesh, contours, sortData ); } );
    std::vector<char> crossed( faces.size() ); // not in a bit set yet, since its neighbor faces share words
    ParallelFor( faces, finders, [&] ( size_t k, FaceCrossingsFinder& finder ) { crossed[k] = finder.find( faces[k] ); } );
    FaceBitSet res( mesh.topology.faceSize() );
    for ( size_t k = 0; k < faces.size(); ++k )
        if ( crossed[k] )
            res.set( faces[k].front().f );
    return res;
}

} //anonymous namespace

Expected<OneMeshContours> offsetSurfaceLineToMeshContours( const Mesh& mesh, const std::vector<MeshTriPoint>& meshTriPoints, float isoValue, SearchPathSettings searchSettings )
{
    return offsetSurfaceLineToMeshContours( mesh, meshTriPoints,
        [isoValue] ( int )
    {
        return isoValue;
    }, searchSettings );
}

Expected<OneMeshContours> offsetSurfaceLineToMeshContours( const Mesh& mesh, const std::vector<MeshTriPoint>& meshTriPoints,
    const std::function<float( int )>& offsetAtPoint, SearchPathSettings searchSettings )
{
    if ( !offsetAtPoint )
    {
        assert( false );
        return {};
    }
    MinMaxf mmOffset;
    for ( int i = 0; i < meshTriPoints.size(); ++i )
        mmOffset.include( std::abs( offsetAtPoint( i ) ) );

    std::vector<int> pivotIndices; // only used if offset is variable
    auto initCutContourRes = surfaceLineToMeshContour( mesh, meshTriPoints, searchSettings, mmOffset.min == mmOffset.max ? nullptr : &pivotIndices );
    if ( !initCutContourRes.has_value() )
        return unexpected( initCutContourRes.error() );

    OneMeshContours initCutContoursVec = OneMeshContours{ std::move( *initCutContourRes ) };

    if ( mmOffset.min == 0.0f && mmOffset.max == 0.0f )
        return initCutContoursVec;

    Mesh meshAfterCut = mesh;
    NewEdgesMap nEM;
    // force fill because we don't really care about intermediate mesh quality
    auto cutRes = cutMesh( meshAfterCut, initCutContoursVec, { .forceFill = true, .new2oldEdgesMap = &nEM } );

    const auto& initCutContours = initCutContoursVec[0];

    // setup start vertices
    VertBitSet startVertices( meshAfterCut.topology.vertSize() );
    HashMap<VertId, float> startVerticesValues; // only used if offset is variable
    std::vector<float> sumLength; // only used if offset is variable
    struct InterInfo
    {
        float length{ 0.0f };
        int startPivot{ -1 };
        int endPivot{ -1 };
    };
    std::vector<InterInfo> singleSegmentInfo; // only used if offset is variable
    if ( !pivotIndices.empty() ) // if offset is variable
    {
        sumLength.resize( pivotIndices.size(), 0.0f );
        singleSegmentInfo.resize( initCutContours.intersections.size() );
        int startPivot = 0;
        int endPivot = 0;
        while ( pivotIndices[startPivot] == -1 )
            startPivot = ( startPivot + 1 ) % int( pivotIndices.size() );
        while ( pivotIndices[endPivot] == -1 )
            endPivot = ( endPivot + 1 ) % int( pivotIndices.size() );
        bool lastSegment = false;
        for ( int i = 0; i < initCutContours.intersections.size(); ++i )
        {
            if ( !lastSegment && pivotIndices[endPivot] <= i )
            {
                startPivot = endPivot;
                endPivot = ( startPivot + 1 ) % int( pivotIndices.size() );
                while ( pivotIndices[endPivot] == -1 )
                    endPivot = ( endPivot + 1 ) % int( pivotIndices.size() );
                if ( endPivot < startPivot )
                    lastSegment = true;
            }
            float length = 0.0f;
            if ( pivotIndices[startPivot] != i )
            {
                int prevI = ( i - 1 + int( singleSegmentInfo.size() ) ) % int( singleSegmentInfo.size() );
                length = ( initCutContours.intersections[i].coordinate - initCutContours.intersections[prevI].coordinate ).length();
            }
            singleSegmentInfo[i].length += length;
            singleSegmentInfo[i].startPivot = startPivot;
            singleSegmentInfo[i].endPivot = endPivot;
            sumLength[startPivot] += length;
        }
    }
    // cutMesh added a new vertex per intersection not in a vertex, except the closing duplicate of the first one
    VertId currentId( mesh.topology.vertSize() );
    const int numUniqueInters = int( initCutContours.intersections.size() ) - ( initCutContours.closed ? 1 : 0 );
    for ( int i = 0; i < numUniqueInters; ++i )
    {
        const auto inter = initCutContours.intersections[i];
        VertId interVertId;
        if ( std::holds_alternative<VertId>( inter.primitiveId ) )
            interVertId = std::get<VertId>( inter.primitiveId );
        else
            interVertId = currentId++;
        startVertices.set( interVertId );
        if ( !pivotIndices.empty() ) // if offset is variable
        {
            auto curSumLength = sumLength[singleSegmentInfo[i].startPivot];
            float ratio = curSumLength > 0.0f ? singleSegmentInfo[i].length / curSumLength : 0.5f;
            float offsetValue = ( 1.0f - ratio ) * offsetAtPoint( singleSegmentInfo[i].startPivot ) + ratio * offsetAtPoint( singleSegmentInfo[i].endPivot );
            startVerticesValues[interVertId] = mmOffset.max - offsetValue;
        }
    }

    // calculate isolines
    VertScalars distances;
    if ( pivotIndices.empty() ) // if offset is static
        distances = computeSurfaceDistances( meshAfterCut, startVertices, FLT_MAX );
    else
        distances = computeSurfaceDistances( meshAfterCut, startVerticesValues, FLT_MAX );
    auto isoLines = extractIsolines( meshAfterCut.topology, distances, mmOffset.max );

    OneMeshContours res;
    res.resize( isoLines.size() );
    for ( int i = 0; i < isoLines.size(); ++i )
    {
        const auto& isoLineI = isoLines[i];
        auto& resI = res[i];
        // an iso-line that reaches the mesh boundary is open: it starts and ends on boundary edges
        assert( isoLineI.size() > 1 );
        resI.closed = isoLineI.front() == isoLineI.back();
        resI.intersections.resize( isoLineI.size() );
        ParallelFor( resI.intersections, [&] ( size_t j )
        {
            const auto& mep = isoLineI[j];
            auto& inter = resI.intersections[j];

            inter.coordinate = meshAfterCut.edgePoint( mep );

            auto newUE = mep.e.undirected();
            if ( newUE < mesh.topology.undirectedEdgeSize() )
                inter.primitiveId = mep.e;
            else
            {
                if ( nEM.splitEdges.test( newUE ) )
                {
                    auto oldE = EdgeId( nEM.map[newUE] );
                    if ( mep.e.odd() )
                        oldE = oldE.sym();
                    inter.primitiveId = oldE;
                }
                else
                {
                    inter.primitiveId = FaceId( nEM.map[newUE] );
                }
            }
        } );
    }
    return res;
}

CutMeshResult cutMesh( Mesh& mesh, const OneMeshContours& contours, const CutMeshParameters& params )
{
    MR_TIMER;
    mesh.invalidateCaches();
    CutMeshResult res;
    if ( params.new2OldMap )
        prepareFacesMap( mesh.topology, *params.new2OldMap );

    // under force fill, the faces where the contours really cross (touching is fine) are bad too, found before the cut
    if ( params.forceFill )
        res.fbsWithContourIntersections = findCrossingFaces( mesh, contours, params.sortData );

    const auto loneFaces = findLoneFaces( mesh.topology, contours );
    const auto firstNewEdge = UndirectedEdgeId( mesh.topology.undirectedEdgeSize() );
    auto preRes = doPreCutMesh( mesh, contours, params.sortData );

    if ( params.new2oldEdgesMap )
    {
        Timer t( "new2oldEdgesMap" );
        for ( int i = 0; i < preRes.paths.size(); ++i )
        {
            for ( int j = 0; j < preRes.paths[i].size(); ++j )
            {
                params.new2oldEdgesMap->map[preRes.paths[i][j].undirected()] = preRes.oldEdgesInfo[i][j].left;
            }
        }
    }

    cutEdgesIntoPieces( mesh, std::move( preRes.edgeData ), contours, params.sortData, params.new2OldMap, params.new2oldEdgesMap );
    FaceBitSet badLoneFaces;
    connectLoneContours( mesh, contours, loneFaces, preRes, firstNewEdge, params.sortData, params.new2oldEdgesMap, badLoneFaces );

    res.fbsWithContourIntersections |= getBadFacesAfterCut( mesh.topology, preRes ) | badLoneFaces;
    if ( !params.forceFill && res.fbsWithContourIntersections.any() )
        return res;

    // find one edge for every hole to fill
    Timer t( "find edge per hole" );
    EdgeBitSet allHoleEdges( mesh.topology.edgeSize() );
    std::vector<EdgeId> holeRepresentativeEdges;
    std::vector<FaceId> oldFaces; // of corresponding holeRepresentativeEdges
    std::vector<EdgeId> pseudoHoleRepresentativeEdges;
    std::vector<FaceId> pseudoOldFaces; // of corresponding pseudoHoleRepresentativeEdges
    const bool needOldFaces = params.new2OldMap || params.new2oldEdgesMap;
    auto addHoleDesc = [&] ( EdgeId e, FaceId oldf, bool pseudofill )
    {
        if ( allHoleEdges.test( e ) )
            return;
        if ( !pseudofill )
        {
            holeRepresentativeEdges.push_back( e );
            if ( needOldFaces )
                oldFaces.push_back( oldf );
        }
        else
        {
            pseudoHoleRepresentativeEdges.push_back( e );
            if ( needOldFaces )
                pseudoOldFaces.push_back( oldf );
        }
        for ( auto ei : leftRing( mesh.topology, e ) )
        {
            [[maybe_unused]] auto v = allHoleEdges.test_set( ei );
            assert( !v );
        }
    };
    bool fillLeft = params.fillPart == CutMeshParameters::FillPart::Both || params.fillPart == CutMeshParameters::FillPart::Left;
    bool fillRight = params.fillPart == CutMeshParameters::FillPart::Both || params.fillPart == CutMeshParameters::FillPart::Right;
    for ( int pathId = 0; pathId < preRes.paths.size(); ++pathId )
    {
        const auto& path = preRes.paths[pathId];
        const auto& oldEdgesInfo = preRes.oldEdgesInfo[pathId];
        for ( int edgeId = 0; edgeId < path.size(); ++edgeId )
        {
            const FaceId oldf = oldEdgesInfo[edgeId].left;
            if ( !oldf.valid() )
                continue;
            const FaceId oldRightF = oldEdgesInfo[edgeId].right;
            bool addLeftDesc = !mesh.topology.left( path[edgeId] ).valid();
            bool addRightDesc = oldRightF && !mesh.topology.right( path[edgeId] ).valid();
            // order does matter because bad holes have tunnels and passes through both left and right
            if ( !fillLeft )
            {
                if ( addRightDesc ) addHoleDesc( path[edgeId].sym(), oldRightF, !fillRight );
                if ( addLeftDesc ) addHoleDesc( path[edgeId], oldf, true ); // pseudo hole should always be second
            }
            else
            {
                if ( addLeftDesc ) addHoleDesc( path[edgeId], oldf, false ); // real hole should alwauys be first
                if ( addRightDesc ) addHoleDesc( path[edgeId].sym(), oldRightF, !fillRight );
            }
        }
    }
    t.finish();

    // prepare in parallel the plan to fill every contour
    auto fillPlans = getPlanarHoleFillPlans( mesh, holeRepresentativeEdges );

    // fill contours
    t.restart( "run TriangulateContourPlans" );
    int numTris = int( pseudoHoleRepresentativeEdges.size() );
    int numEdges = 0;
    for ( const auto& plan : fillPlans )
    {
        numTris += plan.numTris;
        numEdges += 2 * int( plan.items.size() );
    }
    const auto expectedTotalTris = mesh.topology.faceSize() + numTris;
    const auto expectedTotalEdges = mesh.topology.edgeSize() + numEdges;

    mesh.topology.faceReserve( expectedTotalTris );
    mesh.topology.edgeReserve( expectedTotalEdges );
    if ( params.new2OldMap )
        params.new2OldMap->reserve( expectedTotalTris );

    for ( size_t i = 0; i < holeRepresentativeEdges.size(); ++i )
    {
        // regenerate fill plan, if the execution of the old one (prepared before filling of other holes)
        // leads to the appearance of multiple edges
        if ( !isFillingMultipleEdgeFree( mesh.topology, fillPlans[i] ) )
            fillPlans[i] = getPlanarHoleFillPlan( mesh, holeRepresentativeEdges[i], false );

        executeTriangulateContourPlan( mesh, holeRepresentativeEdges[i], fillPlans[i],
            needOldFaces ? oldFaces[i] : FaceId{}, params.new2OldMap, params.new2oldEdgesMap );
    }

    auto pseudoFill = [&] ( EdgeId eid, FaceId of )
    {
        auto nf = mesh.topology.addFaceId();
        mesh.topology.setLeft( eid, nf );
        if ( params.new2OldMap )
        {
            assert( of.valid() );
            params.new2OldMap->autoResizeSet( nf, of );
        }
    };
    for ( size_t i = 0; i < pseudoHoleRepresentativeEdges.size(); ++i )
    {
        pseudoFill( pseudoHoleRepresentativeEdges[i], needOldFaces ? pseudoOldFaces[i] : FaceId{} );
    }

    assert( mesh.topology.faceSize() == expectedTotalTris );
    if ( params.new2OldMap )
        assert( params.new2OldMap->size() == ( numTris != 0 ? expectedTotalTris : mesh.topology.lastValidFace() + 1 ) );

    res.resultCut = std::move( preRes.paths );

    return res;
}

Expected<FaceBitSet> cutMeshByContour( Mesh& mesh, const Contour3f& contour, const AffineXf3f& xf )
{
    return cutMeshByContours( mesh, { contour }, xf );
}

Expected<FaceBitSet> cutMeshByContours( Mesh& mesh, const Contours3f& contours, const AffineXf3f& xf )
{
    MR_TIMER;
    if ( mesh.topology.faceSize() <= 0 )
        return unexpected( "Mesh is empty" );

    std::vector<Expected<OneMeshContour>> maybeOneMeshContours( contours.size() );
    ParallelFor( contours, [&]( size_t ic )
    {
        const auto & contour = contours[ic];
        std::vector<MeshTriPoint> surfaceLine( contour.size() );
        ParallelFor( surfaceLine, [&] ( size_t i )
        {
            auto proj = findProjection( xf( contour[i] ), mesh );
            surfaceLine[i] = proj.mtp;
        } );
        maybeOneMeshContours[ic] = surfaceLineToMeshContour( mesh, surfaceLine );
    } );

    std::vector<OneMeshContour> oneMeshContours( contours.size() );
    for( size_t ic = 0; ic < contours.size(); ++ic )
    {
        if ( !maybeOneMeshContours[ic] )
            return unexpected( std::move( maybeOneMeshContours[ic].error() ) );
        oneMeshContours[ic] = std::move( *maybeOneMeshContours[ic] );
    }

    auto cutRes = cutMesh( mesh, oneMeshContours );
    if ( !cutRes.fbsWithContourIntersections.none() )
        return unexpected( "Cannot cut mesh because of contour self intersections" );
    auto sideFbv = fillContourLeftByGraphCut( mesh.topology, cutRes.resultCut, edgeLengthMetric( mesh ) );
    return sideFbv;
}

Expected<std::vector<MR::EdgePath>> cutMeshByProjection( Mesh& mesh, const Contours3f& contours, const CutByProjectionSettings& settings )
{
    if ( settings.direction == Vector3f() )
    {
        assert( false );
        return unexpected( "Invalid direction" );
    }

    OneMeshContours cutConts( contours.size() );
    std::string error;
    auto rayPrecomp = IntersectionPrecomputes<float>( settings.direction );
    tbb::task_group_context ctx;
    ParallelFor( contours, [&] ( size_t i )
    {
        if ( ctx.is_group_execution_cancelled() )
            return; // fast exit before projecting
        auto mtps = std::vector<MeshTriPoint>( contours[i].size() );
        ParallelFor( mtps, [&] ( size_t j )
        {
            if ( ctx.is_group_execution_cancelled() )
                return; // stop projecting if failed
            auto intRes = rayMeshIntersect( mesh, Line3f( settings.cont2mesh ? ( *settings.cont2mesh )( contours[i][j] ) : contours[i][j], settings.direction ), 
                0, FLT_MAX, &rayPrecomp );
            if ( !intRes )
            {
                if ( ctx.cancel_group_execution() )
                    error = "Some contour points have missed the mesh";
                return;
            }
            mtps[j] = intRes.mtp;
        } );
        if ( ctx.is_group_execution_cancelled() )
            return; // do not continue if projecting failed
        auto cutContRes = surfaceLineToMeshContour( mesh, mtps,
            [&] ( const MeshTriPoint& start, const MeshTriPoint& end, int, int )->Expected<SurfacePath>
        {
            auto plPoint = mesh.triPoint( start ) - settings.direction;
            auto ccwPath = trackSection( mesh, start, end, plPoint, true );
            auto cwPath = trackSection( mesh, start, end, plPoint, false );
            if ( ccwPath.has_value() && cwPath.has_value() )
            {
                auto ccwL = surfacePathLength( mesh, *ccwPath );
                auto cwL = surfacePathLength( mesh, *cwPath );
                if ( ccwL < cwL )
                    return ccwPath;
                else
                    return cwPath;
            }
            else if ( ccwPath.has_value() )
                return ccwPath;
            else if ( cwPath.has_value() )
                return cwPath;
            else
            {
                auto locRes = computeGeodesicPath( mesh, start, end );
                if ( !locRes.has_value() )
                    return unexpected( toString( locRes.error() ) );
                return *locRes;
            }
        } );
        if ( !cutContRes.has_value() )
        {
            if ( ctx.cancel_group_execution() )
                error = std::move( cutContRes.error() );
            return;
        }
        cutConts[i] = std::move( *cutContRes );
    } );
    if ( ctx.is_group_execution_cancelled() )
        return unexpected( std::move( error ) );

    auto cutRes = cutMesh( mesh, cutConts );
    if ( cutRes.fbsWithContourIntersections.any() )
        return unexpected( "Some projected contours have self-intersections" );
    return std::move( cutRes.resultCut );
}

} //namespace MR
