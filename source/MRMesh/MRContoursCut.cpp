#include "MRContoursCut.h"
#include "MRMesh.h"
#include "MRPolyline.h"
#include "MRTriangleIntersection.h"
#include "MRTriMath.h"
#include "MRMeshTopology.h"
#include "MRMeshDelone.h"
#include "MR2DContoursTriangulation.h"
#include "MRCutTrianglePredicates.h"
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
#include <atomic>
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

using EdgeData = Vector<EdgeIntersectionData>;

// the faces of the old mesh on both sides of a path edge: the same face for a path edge inside it,
// the faces of an edge of the old mesh for a path edge along it
struct PathsEdgeIndex
{
    FaceId left;
    FaceId right;
};

using PathsOldFaces = Vector<PathsEdgeIndex>; // path edge i of contour c at the index of its intersection i in the contours
using EdgeDataMap = ParallelHashMap<UndirectedEdgeId, EdgeData>;
struct PreCutResult
{
    EdgeDataMap edgeData;
    std::vector<EdgePath> paths;
    PathsOldFaces oldEdgesInfo;
    const Vector<int>* offsets = nullptr; // of the contours
    HashMap<FaceId, std::array<EdgeId, 3>> vertFaces; // the old faces of path edges at mesh vertices, removed by doPreCutMesh, and their edges

    // the old faces of the path edges of contour c
    const PathsEdgeIndex* oldFaces( int c ) const { return oldEdgesInfo.data() + ( *offsets )[c]; }
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

/// prepares precise-coordinates for the corners of a triangle of the other mesh
std::array<PreciseVertCoords, 3> preciseOtherTri( FaceId t, const SortIntersectionsData& sortData )
{
    const auto vs = sortData.otherMesh.topology.getTriVerts( t );
    return { preciseOtherVert( vs[0], sortData ), preciseOtherVert( vs[1], sortData ), preciseOtherVert( vs[2], sortData ) };
}

/// the exact definition of intersection i of contour c inside a triangle of the mesh with given edges, counterclockwise,
/// and tri[k] the precise coordinates of org( edges[k] ) before the cut: a corner, a side of the triangle crossing a triangle of the other mesh,
/// or an edge of the other mesh crossing the triangle
PreciseCutPoint preciseCutPoint( const Mesh& mesh, const OneMeshContours& contours, int c, int i,
    const std::array<EdgeId, 3>& edges, const std::array<PreciseVertCoords, 3>& tri, const SortIntersectionsData& sortData )
{
    PreciseCutPoint res;
    const auto& prim = contours.intersection( c, i );
    const auto& et = sortData.contours[c][i];
    const auto& otherTopology = sortData.otherMesh.topology;
    if ( const auto pe = prim.edge() )
    {
        for ( int k = 0; k < 3; ++k )
            if ( edges[k].undirected() == pe.undirected() )
                res.seg = { tri[k], tri[( k + 1 ) % 3] };
        res.tri = preciseOtherTri( et.tri(), sortData );
    }
    else if ( prim.isFace() )
    {
        // the third vertices of the triangles of the edge tell apart the crossings of a multiple edge
        const EdgeId e = et.edge;
        res.seg = { preciseOtherVert( otherTopology.org( e ), sortData ), preciseOtherVert( otherTopology.dest( e ), sortData ) };
        res.tri = tri;
        if ( otherTopology.left( e ) && otherTopology.right( e ) )
            res.wedge = { preciseOtherVert( otherTopology.getLeftTriVerts( e )[2], sortData ),
                preciseOtherVert( otherTopology.getLeftTriVerts( e.sym() )[2], sortData ) };
    }
    else
        res.seg[0] = preciseOwnVert( prim.vert(), mesh, sortData );
    return res;
}

// whether path edge i of the contour (from intersection i to i + 1) is a new edge: neither an edge of the old mesh,
// nor the same edge as another path edge where the contour touches a point from a mesh vertex and goes back
bool isNewPathEdge( const OneMeshContourView& contour, int i )
{
    const auto& inters = contour.intersections;
    const auto vert = inters[i].vert();
    const auto nextVert = inters[i + 1].vert();
    if ( vert && nextVert )
        return false;
    if ( !vert && nextVert && i > 0 && inters[i - 1].vert() == nextVert )
        return false;
    return !( vert && contour.closed && i + 2 == inters.size() && inters[1].samePrimitive( inters[i] ) );
}

PreCutResult doPreCutMesh( Mesh& mesh, const OneMeshContours& contours, const SortIntersectionsData* sortData )
{
    MR_TIMER;
    auto& topology = mesh.topology;

    int intersectedEdges = 0;
    for ( const auto& in : contours.intersections )
        if ( in.isEdge() )
            ++intersectedEdges;
    reserveForCut( mesh, mesh, contours );

    PreCutResult res;
    res.paths.resize( contours.size() );
    res.oldEdgesInfo.resize( contours.intersections.size() );
    res.offsets = &contours.offsets;
    res.edgeData.reserve( size_t( intersectedEdges ) );
    const auto oldEdgesSize = topology.edgeSize();

    // find the old face of every path edge before the topology changes, on this thread, which then has the records of the cut edges
    // in its cache for cutOneEdge; the faces of path edges at mesh vertices are removed here (the others by cutOneEdge or
    // connectLoneContours), and their edges show where to splice the path edges around the vertices
    auto& vertFaces = res.vertFaces;
    const auto& inters = contours.intersections;
    for ( int c = 0; c < contours.size(); ++c )
    {
        for ( int k = contours.offsets[c]; k + 1 < contours.offsets[c + 1]; ++k )
        {
            const auto& a = inters[k];
            const auto& b = inters[k + 1];
            auto& oldFaces = res.oldEdgesInfo[k];
            const auto va = a.vert();
            const auto vb = b.vert();
            if ( va && vb )
            {
                // the path edge is an edge of the old mesh
                const auto e = topology.findEdge( va, vb );
                assert( e );
                if ( !e )
                    continue;
                oldFaces = { topology.left( e ), topology.right( e ) };
            }
            else if ( const auto pe = ( va ? b : a ).edge() )
            {
                // a path edge leaves an edge point into the left face of the edge and arrives at it from the right face
                const auto e = va ? pe.sym() : pe;
                const auto v = va ? va : vb;
                // precondition: a mesh vertex at the other end is the third vertex of that face, so the contour does not fold back
                // at the edge; a fold puts the path edge in the other face, where Release still gets a valid topology
                // (a mesh vertex that is no corner of these faces is not handled)
                const bool fold = v && ( !topology.left( e ) || topology.dest( topology.next( e ) ) != v );
                assert( !fold );
                oldFaces.left = oldFaces.right = topology.left( fold ? e.sym() : e );
            }
            else
                oldFaces.left = oldFaces.right = ( va ? b : a ).face();
            if ( oldFaces.left && ( va || vb ) )
            {
                auto [it, inserted] = vertFaces.try_emplace( oldFaces.left );
                if ( inserted )
                    topology.getTriEdges( oldFaces.left, it->second[0], it->second[1], it->second[2] );
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
            const auto tri = preciseOwnTri( fv, mesh, *sortData );
            preds->reset( tri );
            PreciseCutPoint corner;
            corner.seg[0] = tri[k];
            preds->addPoint( VertId( 0 ), corner );
            preds->addPoint( VertId( 1 ), preciseCutPoint( mesh, contours, c, i, fe, tri, *sortData ) );
            int n = 2;
            for ( auto y = topology.next( x0 ); y >= oldEdgesSize; y = topology.next( y ) )
            {
                const auto [yc, yi] = spliceEnds.at( y );
                preds->addPoint( VertId( n++ ), preciseCutPoint( mesh, contours, yc, yi, fe, tri, *sortData ) );
            }
            exact = !preds->failed();
        }
        const auto& pv = mesh.points[v];
        const auto normal = cross( mesh.points[fv[( k + 1 ) % 3]] - pv, mesh.points[fv[( k + 2 ) % 3]] - pv );
        const auto p = contours.intersection( c, i ).coordinate - pv;
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
                    if ( !( dot( cross( contours.intersection( yc, yi ).coordinate - pv, p ), normal ) > 0 ) )
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
        // the new path edges of the contour at once, taken in the order makeEdge() would make them
        int newPathEdges = 0;
        for ( int i = 0; i + 1 < inContour.size(); ++i )
            if ( isNewPathEdge( contours[contourId], i ) )
                ++newPathEdges;
        auto newEdge = topology.makeEdges( newPathEdges );
        const auto* faces = res.oldEdgesInfo.data() + contours.offsets[contourId];
        auto& path = res.paths[contourId];
        path.resize( inContour.size() - 1 );
        const bool closed = contours[contourId].closed;
        for ( int i = 0; i < inContour.size(); ++i )
        {
            const auto& inter = inContour[i];
            const auto vert = inter.vert();
            // the path edges to and from this intersection
            const EdgeId in = i > 0 ? path[i - 1].sym() : EdgeId{};
            EdgeId out;
            if ( i + 1 < inContour.size() )
            {
                const auto nextVert = inContour[i + 1].vert();
                if ( vert && nextVert )
                    out = topology.findEdge( vert, nextVert );
                else if ( isNewPathEdge( contours[contourId], i ) )
                {
                    out = newEdge;
                    newEdge += 2;
                }
                else if ( !vert )
                    out = in; // the contour touches this point from a mesh vertex and goes back along the same edge
                else
                    out = path.front().sym(); // the same at the first point of a closed contour (out == in for x, v, x)
                path[i] = out;
            }
            if ( vert )
            {
                // a path edge with an origin is already in place: an edge of the old mesh, or the edge back from a touched point
                if ( out && !topology.org( out ) )
                    spliceAtVert( vert, faces[i].left, out, contourId, i + 1 );
                if ( in && !topology.org( in ) )
                    spliceAtVert( vert, faces[i - 1].left, in, contourId, i - 1 );
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
            if ( const auto e = inter.edge() )
            {
                auto& edgeData = res.edgeData[e.undirected()];
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

// a face of the mesh that contours go inside, with its sides before the cut
struct CutFace
{
    FaceId f;
    std::array<EdgeId, 3> edges; // counterclockwise, before the cut
    ThreeVertIds verts; // verts[i] = org( edges[i] )
};

// face f with its sides before the cut changes their topology (the faces that doPreCutMesh removed have their edges in vertFaces)
CutFace cutFace( const MeshTopology& topology, FaceId f, const HashMap<FaceId, std::array<EdgeId, 3>>* vertFaces = nullptr )
{
    CutFace res{ f, vertFaces ? getAt( *vertFaces, f ) : std::array<EdgeId, 3>{} };
    if ( !res.edges[0] )
        topology.getTriEdges( f, res.edges[0], res.edges[1], res.edges[2] );
    for ( int k = 0; k < 3; ++k )
        res.verts[k] = topology.org( res.edges[k] );
    return res;
}

// finds the faces of lone contours, sorted, before the cut changes their topology;
// a lone contour is inside one face crossing none of its edges (e.g. where a tip of the other mesh pokes through it),
// so it is an island in the hole of its face, or a slit if it is open
Vector<CutFace> findLoneFaces( const MeshTopology& topology, const OneMeshContours& contours )
{
    Vector<FaceId> faces;
    for ( const auto& c : contours )
    {
        if ( c.intersections.size() > 1 && std::all_of( c.intersections.begin(), c.intersections.end(),
            [] ( const OneMeshIntersection& i ) { return i.isFace(); } ) )
            faces.push_back( c.intersections.front().face() );
    }
    std::sort( faces.begin(), faces.end() );
    faces.erase( std::unique( faces.begin(), faces.end() ), faces.end() );
    Vector<CutFace> res( faces.size() );
    for ( int i = 0; i < faces.size(); ++i )
        res[i] = cutFace( topology, faces[i] );
    return res;
}

// the unit normal of a triangle of the mesh, or zero if its corners are on one line: the corners in doubles give a normal free of the rounding
// of the points inside the triangle, which a sum over them would add; normalized before the conversion to float, which could flush it
// to zero or overflow at extreme scales
Vector3f cornersNormal( const Mesh& mesh, const ThreeVertIds& verts )
{
    const auto& p = mesh.points;
    const auto normal = dirDblArea( Triangle3d{ Vector3d( p[verts[0]] ), Vector3d( p[verts[1]] ), Vector3d( p[verts[2]] ) } );
    return normal == Vector3d() ? Vector3f() : Vector3f( normal.normalized() );
}

// the exact definitions of the vertices on the hole loops of the faces that contours go inside, consistent with the boolean,
// if sortData is given
struct ExactCutPoints
{
    const Mesh& mesh;
    const OneMeshContours& contours;
    const SortIntersectionsData* sortData;
    // the contour point of every vertex of the paths inside the faces, as {contour, point}
    HashMap<VertId, std::pair<int, int>> vertInters;

    // the definition of vertex v on the loops of face cf with precise corners tri
    PreciseCutPoint point( const CutFace& cf, const std::array<PreciseVertCoords, 3>& tri, VertId v ) const
    {
        const auto it = vertInters.find( v );
        if ( it == vertInters.end() )
        {
            PreciseCutPoint res;
            res.seg[0] = preciseOwnVert( v, mesh, *sortData ); // a corner of the face
            return res;
        }
        return preciseCutPoint( mesh, contours, it->second.first, it->second.second, cf.edges, tri, *sortData );
    }
};

// the hole loops through the given edges, skipping the edges with a face on the left and those already in loops, which it marks in inLoops
EdgeLoops holeLoops( const MeshTopology& topology, const EdgePath& edges, EdgeBitSet& inLoops )
{
    EdgeLoops res;
    for ( auto e : edges )
    {
        if ( topology.left( e ) || inLoops.test( e ) )
            continue;
        auto& loop = res.emplace_back();
        for ( auto le : leftRing( topology, e ) )
        {
            inLoops.set( le );
            loop.push_back( le );
        }
    }
    return res;
}

// the holes of faces that contours go inside
struct FaceHoles
{
    Vector<EdgePath> edges; // the path edges inside every face, with the face on the left, and a side of it if it was removed here
    Vector<EdgeLoops> loops; // the boundaries of the hole of every face
    BitSet removedHere; // the faces that no other contour removed, to put back if their plans fail
};

// removes the faces that no other contour removed, and finds the holes of all of them; sets the contour points at the ends of the path edges
// in exact if it has sortData
FaceHoles makeFaceHoles( MeshTopology& topology, const Vector<CutFace>& faces, const PreCutResult& preRes, UndirectedEdgeId firstNewEdge,
    ExactCutPoints& exact )
{
    FaceHoles res{ Vector<EdgePath>( faces.size() ), Vector<EdgeLoops>( faces.size() ), BitSet( faces.size() ) };
    for ( int i = 0; i < preRes.paths.size(); ++i )
    {
        for ( int j = 0; j < preRes.paths[i].size(); ++j )
        {
            const auto f = preRes.oldFaces( i )[j].left;
            const auto it = std::lower_bound( faces.begin(), faces.end(), f, [] ( const CutFace& cf, FaceId g ) { return cf.f < g; } );
            if ( it == faces.end() || it->f != f )
                continue;
            auto& edges = res.edges[it - faces.begin()];
            const auto e = preRes.paths[i][j];
            edges.push_back( e );
            if ( e.undirected() >= firstNewEdge ) // a new edge is inside the face, while an old one has it on the left only
                edges.push_back( e.sym() );
            if ( !exact.sortData )
                continue;
            exact.vertInters[topology.org( e )] = { i, j };
            exact.vertInters[topology.dest( e )] = { i, j + 1 };
        }
    }

    EdgeBitSet inLoops( topology.edgeSize() );
    for ( int i = 0; i < faces.size(); ++i )
    {
        const auto& cf = faces[i];
        if ( topology.hasFace( cf.f ) ) // no other contour removed it
        {
            topology.setLeft( cf.edges[0], {} );
            res.edges[i].push_back( cf.edges[0] );
            res.removedHere.set( i );
        }
        res.loops[i] = holeLoops( topology, res.edges[i], inLoops );
    }
    return res;
}

// adds the chords that split the holes of the faces (bounded by all their loops together) in monotone parts, on the exact predicates
// consistent with the boolean if exact has sortData; the faces where that fails are added to badFaces, and put back if removed here
void addMonotoneChords( Mesh& mesh, const Vector<CutFace>& faces, const Vector<EdgeLoops>& faceLoops, const BitSet& removedHere,
    const ExactCutPoints& exact, NewEdgesMap* new2OldEdgeMap, FaceBitSet& badFaces )
{
    // the faces are independent, so plan them in parallel
    Vector<std::optional<HoleFillPlan>> plans( faces.size() );
    ParallelFor( plans, [&] ( size_t i )
    {
        const auto& cf = faces[i];
        const auto& loops = faceLoops[i];
        if ( exact.sortData )
        {
            const auto tri = preciseOwnTri( cf.verts, mesh, *exact.sortData );
            plans[i] = PlanarTriangulation::getMonotonePlan( mesh, loops, tri, [&] ( VertId v ) { return exact.point( cf, tri, v ); } );
            return;
        }
        // the loops go counterclockwise around the normal of the face, of which only the dominant component and its sign matter
        const auto normal = cornersNormal( mesh, cf.verts );
        // corners on one line give no normal, so no plane to connect the islands in: the plan stays empty, and the face is reported
        if ( normal == Vector3f() )
            return;
        plans[i] = PlanarTriangulation::getMonotonePlan( mesh, loops, normal );
    } );

    for ( int i = 0; i < faces.size(); ++i )
    {
        auto& plan = plans[i];
        // every island or slit needs a chord to the rest of its face, so a plan without any leaves them unconnected
        // (as where the cut of a side has already filled the face again, ending an open contour on that side)
        if ( plan && !plan->items.empty() && isFillingMultipleEdgeFree( mesh.topology, *plan ) )
            executeTriangulateContourPlan( mesh, faceLoops[i].front().front(), *plan, faces[i].f, nullptr, new2OldEdgeMap );
        else
        {
            badFaces.autoResizeSet( faces[i].f );
            // the holes to fill are found by the cut paths, and the old sides of the face have none, so the face is put back;
            // no chord of it was added, so its old sides still make its ring
            if ( removedHere.test( i ) )
                mesh.topology.setLeft( faces[i].edges[0], faces[i].f );
        }
    }
}

// connects the islands of lone contours with the rest of the holes of their faces by the chords that split the holes in monotone parts,
// found on the exact predicates consistent with the boolean if sortData is given; the faces where that fails are added to badFaces
// and stay in the mesh unless another contour removed them
void connectLoneContours( Mesh& mesh, const OneMeshContours& contours, const Vector<CutFace>& loneFaces, const PreCutResult& preRes,
    UndirectedEdgeId firstNewEdge, const SortIntersectionsData* sortData, NewEdgesMap* new2OldEdgeMap, FaceBitSet& badFaces )
{
    if ( loneFaces.empty() )
        return;
    MR_TIMER;

    ExactCutPoints exact{ mesh, contours, sortData };
    const auto holes = makeFaceHoles( mesh.topology, loneFaces, preRes, firstNewEdge, exact );
    addMonotoneChords( mesh, loneFaces, holes.loops, holes.removedHere, exact, new2OldEdgeMap, badFaces );
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
    EdgeId baseEdge = contours.intersection( intInfo.contourId, intInfo.intersectionId ).edge();

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
        Vector<double> dotProds( edgeData.size() );
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

// the edge to cut: oriented as in the intersection of edgeData that was first before sorting
EdgeId baseEdgeOf( const EdgeData& edgeData, const OneMeshContours& contours )
{
    const auto& intInfo = std::find_if( edgeData.begin(), edgeData.end(), [] ( const auto& data )
    {
        return data.beforeSortIndex == 0;
    } )->interOnEdge;
    return contours.intersection( intInfo.contourId, intInfo.intersectionId ).edge();
}

// the path edges at the new vertex of i-th intersection on the left and on the right of baseEdge
std::pair<EdgeId, EdgeId> leftRightPathEdges( const MeshTopology& topology, EdgeId baseEdge,
    const EdgeData& edgeData, int i, const OneMeshContours& contours )
{
    const auto& interIndex = edgeData[i].interOnEdge;
    const bool isBaseSym = contours.intersection( interIndex.contourId, interIndex.intersectionId ).edge().sym() == baseEdge;
    EdgeId baseLeft = edgeData[i].orgEdgeInLeftTri;
    EdgeId baseRight = baseLeft.valid() ? topology.next( baseLeft ) : topology.edgePerVertex()[edgeData[i].newVert];
    if ( baseLeft == baseRight )
        baseRight = EdgeId{};
    return isBaseSym ? std::pair{ baseRight, baseLeft } : std::pair{ baseLeft, baseRight };
}

// whether all path edges at the new vertices of the edge are on its left (first), on its right (second)
std::pair<bool, bool> allLeftRightOnly( const MeshTopology& topology, EdgeId baseEdge,
    const EdgeData& edgeData, const OneMeshContours& contours )
{
    bool isAllLeftOnly = true;
    bool isAllRightOnly = true;
    for ( int i = 0; i < edgeData.size(); ++i )
    {
        auto [leftEdge, rightEdge] = leftRightPathEdges( topology, baseEdge, edgeData, i, contours );
        if ( rightEdge.valid() )
            isAllLeftOnly = false;
        if ( leftEdge.valid() )
            isAllRightOnly = false;
    }
    return { isAllLeftOnly, isAllRightOnly };
}

// maps the pieces of the edge to it
void mapEdgePieces( NewEdgesMap& new2OldEdgeMap, EdgeId baseEdge, EdgeId e0,
    const EdgeData& edgeData, const OneMeshContours& contours )
{
    new2OldEdgeMap.splitEdges.autoResizeSet( e0.undirected() );
    new2OldEdgeMap.map[e0.undirected()] = baseEdge;
    for ( int i = 0; i + 1 < edgeData.size(); ++i )
    {
        const auto& interIndex = edgeData[i].interOnEdge;
        const bool isBaseSym = contours.intersection( interIndex.contourId, interIndex.intersectionId ).edge().sym() == baseEdge;
        const EdgeId piece = e0 + 2 * ( i + 1 );
        new2OldEdgeMap.map[piece.undirected()] = isBaseSym ? baseEdge.sym() : baseEdge;
        new2OldEdgeMap.splitEdges.autoResizeSet( piece.undirected() );
    }
}

// removes the faces of the edge, detaches it from its origin and makes its other pieces,
// returns the first of them (at the origin), the next ones are the following edges from makeEdges
EdgeId startCutOneEdge( MeshTopology& topology, EdgeId baseEdge,
    const EdgeData& edgeData, const OneMeshContours& contours, NewEdgesMap* new2OldEdgeMap )
{
    topology.setLeft( baseEdge, FaceId{} );
    topology.setLeft( baseEdge.sym(), FaceId{} );

    EdgeId ePrev = topology.prev( baseEdge );
    if ( ePrev != baseEdge )
        topology.splice( ePrev, baseEdge );
    // baseEdge now becomes the last piece of split edge
    const EdgeId e0 = topology.makeEdges( int( edgeData.size() ) );
    if ( ePrev != baseEdge )
        topology.splice( ePrev, e0 );

    if ( new2OldEdgeMap )
        mapEdgePieces( *new2OldEdgeMap, baseEdge, e0, edgeData, contours );
    return e0;
}

// connects the pieces of the edge with the path edges at its new vertices;
// touches only the pieces and the rings of the new vertices, so different edges can be finished in parallel
void finishCutOneEdge( MeshTopology& topology, EdgeId baseEdge, EdgeId e0,
    const EdgeData& edgeData, const OneMeshContours& contours )
{
    for ( int i = 0; i < edgeData.size(); ++i )
    {
        auto [leftEdge, rightEdge] = leftRightPathEdges( topology, baseEdge, edgeData, i, contours );
        const EdgeId bot = e0 + 2 * i;
        const EdgeId top = i + 1 < edgeData.size() ? bot + 2 : baseEdge;
        connectEdges( topology, bot, top, leftEdge, rightEdge );
    }
}

// cuts one edge and connects all intersecting contours with pieces
void cutOneEdge( Mesh& mesh,
                 const EdgeData& edgeData, const OneMeshContours& contours,
                 FaceMap* new2OldMap, NewEdgesMap* new2OldEdgeMap )
{
    assert( !edgeData.empty() );
    const EdgeId baseEdge = baseEdgeOf( edgeData, contours );

    // will need this to restore lost face on first or last intersection (only for open contours)
    FaceId oldLeft = mesh.topology.left( baseEdge );
    FaceId oldRight = mesh.topology.left( baseEdge.sym() );

    const auto [isAllLeftOnly, isAllRightOnly] = allLeftRightOnly( mesh.topology, baseEdge, edgeData, contours );
    const EdgeId e0 = startCutOneEdge( mesh.topology, baseEdge, edgeData, contours, new2OldEdgeMap );
    finishCutOneEdge( mesh.topology, baseEdge, e0, edgeData, contours );

    // fix triangle if this was last or first
    if ( isAllLeftOnly && oldRight.valid() )
        triangulateContour( mesh, baseEdge.sym(), oldRight, new2OldMap, new2OldEdgeMap );
    if ( isAllRightOnly && oldLeft.valid() )
        triangulateContour( mesh, baseEdge, oldLeft, new2OldMap, new2OldEdgeMap );
}

// this function cut mesh edge and connects it with result path,
// after it each path edge left and right faces are invalid (they are removed)
void cutEdgesIntoPieces( Mesh& mesh,
                         EdgeDataMap&& edgeData, const OneMeshContours& contours,
                         const SortIntersectionsData* sortData,
                         FaceMap* new2OldMap, NewEdgesMap* new2OldEdgeMap )
{
    MR_TIMER;
    struct EdgeCut
    {
        const EdgeData* data = nullptr;
        EdgeId base, e0;
    };
    // the cuts in the order of the iteration over edgeData
    Vector<int> subOffsets( edgeData.subcnt() + 1, 0 );
    for ( size_t i = 0; i < edgeData.subcnt(); ++i )
        edgeData.with_submap( i, [&] ( const EdgeDataMap::EmbeddedSet& subSet ) { subOffsets[i + 1] = subOffsets[i] + int( subSet.size() ); } );
    Vector<EdgeCut> cuts( subOffsets.back() );

    // sort each edge intersections in parallel, and find whether some face will be triangulated
    std::atomic<bool> triangulate{ false };
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, edgeData.subcnt(), 1 ),
        [&] ( const tbb::blocked_range<size_t>& range )
    {
        assert( range.begin() + 1 == range.end() );
        for ( size_t i = range.begin(); i != range.end(); ++i )
        {
            edgeData.with_submap( i, [&] ( const EdgeDataMap::EmbeddedSet& subSet )
            {
                int k = subOffsets[i];
                // const_cast here is safe, we don't write to map, just sort internal data
                for ( auto& edgeInfo : const_cast< EdgeDataMap::EmbeddedSet& >( subSet ) )
                {
                    auto& cut = cuts[k++];
                    cut.data = &edgeInfo.second;
                    cut.base = baseEdgeOf( edgeInfo.second, contours );
                    if ( !triangulate.load( std::memory_order_relaxed ) )
                    {
                        auto [l, r] = allLeftRightOnly( mesh.topology, cut.base, edgeInfo.second, contours );
                        if ( l || r )
                            triangulate.store( true, std::memory_order_relaxed );
                    }
                    sortEdgeInfo( mesh, contours, edgeInfo.second, sortData );
                }
            } );
        }
    } );

    if ( triangulate )
    {
        // triangulation makes edges and faces in between, so cut all in order
        for ( const auto& c : cuts )
            cutOneEdge( mesh, *c.data, contours, new2OldMap, new2OldEdgeMap );
        return;
    }

    auto& topology = mesh.topology;
    FaceBitSet faces( topology.faceSize() );
    for ( const auto& c : cuts )
    {
        if ( auto f = topology.left( c.base ) )
            faces.set( f );
        if ( auto f = topology.right( c.base ) )
            faces.set( f );
    }
    topology.deleteFacesKeepEdges( faces );

    // all other pieces at once, with the same ids as from cutOneEdge one edge after another
    int numPieces = 0;
    for ( auto& c : cuts )
    {
        c.e0 = EdgeId( 2 * numPieces ); // relative to the first piece for now
        numPieces += int( c.data->size() );
    }
    const EdgeId firstPiece = topology.makeEdges( numPieces );

    // detach each edge from its origin, putting its first piece there instead; the edges from one vertex in their order,
    // different vertices in parallel
    struct VertCut
    {
        VertId v;
        int cut = 0;
        auto operator<=>( const VertCut& ) const = default;
    };
    Vector<VertCut> byVert( cuts.size() );
    ParallelFor( cuts, [&] ( size_t i )
    {
        cuts[i].e0 += int( firstPiece );
        byVert[i] = { topology.org( cuts[i].base ), int( i ) };
    } );
    tbb::parallel_sort( byVert.begin(), byVert.end() );
    ParallelFor( byVert, [&] ( size_t i )
    {
        if ( i > 0 && byVert[i - 1].v == byVert[i].v )
            return;
        for ( size_t j = i; j < byVert.size() && byVert[j].v == byVert[i].v; ++j )
        {
            const auto& c = cuts[byVert[j].cut];
            EdgeId ePrev = topology.prev( c.base );
            if ( ePrev != c.base )
            {
                topology.splice( ePrev, c.base );
                topology.splice( ePrev, c.e0 );
            }
        }
    } );
    if ( new2OldEdgeMap )
        for ( const auto& c : cuts )
            mapEdgePieces( *new2OldEdgeMap, c.base, c.e0, *c.data, contours );

    ParallelFor( cuts, [&] ( size_t i )
    {
        finishCutOneEdge( topology, cuts[i].base, cuts[i].e0, *cuts[i].data, contours );
    } );
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
            const auto oldf = preRes.oldFaces( pathId )[edgeId].left;
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

// segment i of contour c, from intersection i to i + 1: a new path edge in old face f
struct FaceSegment
{
    FaceId f;
    int c = -1;
    int i = -1;
};

// the buffers of the search in one face, kept for the next face of the thread
struct FaceCrossingsBuffers
{
    Vector<std::pair<VertId, int>> keys; // the vertex of every end of a segment and the end, 2 * segment + k
    Vector<int> defs; // an end at every point
    Vector<TwoVertIds> ends; // the points of every segment
    VertCoords xyz; // the coordinates of the points without sortData
    std::optional<CutTrianglePredicates> preds; // made on the first face with sortData
    std::unique_ptr<PlanarTriangulation::ISweepLineCache> cache; // made on the first face that needs it

    // the cache of the sweep, which only a face of more segments than are tested in pairs needs, or one without sortData,
    // whose projected points it keeps
    PlanarTriangulation::ISweepLineCache* sweepCache()
    {
        if ( !cache )
            cache = PlanarTriangulation::makeSweepLineCache();
        return cache.get();
    }
};

// whether two segments of one face cross: with sortData, by the exact predicates consistent with the boolean, a failure of which makes
// the face bad too (they fail only on the points they compare, so a degenerate face may be bad with many segments for the sweep and not
// with a few tested in pairs); without it, by the float coordinates projected along the axis most aligned with the normal of the face,
// and a face with its corners on one line is bad
bool crossesInFace( FaceCrossingsBuffers& buf, const Mesh& mesh, const OneMeshContours& contours, const PreCutResult& preRes,
    const SortIntersectionsData* sortData, std::span<const FaceSegment> segs )
{
    // one id per vertex, so that segments touching at a common end are never tested
    buf.keys.clear();
    for ( int s = 0; s < int( segs.size() ); ++s )
    {
        const auto e = preRes.paths[segs[s].c][segs[s].i];
        buf.keys.emplace_back( mesh.topology.org( e ), 2 * s );
        buf.keys.emplace_back( mesh.topology.dest( e ), 2 * s + 1 );
    }
    std::sort( buf.keys.begin(), buf.keys.end() );
    buf.defs.clear();
    buf.ends.resize( segs.size() );
    for ( size_t x = 0; x < buf.keys.size(); ++x )
    {
        if ( x == 0 || buf.keys[x].first != buf.keys[x - 1].first )
            buf.defs.push_back( buf.keys[x].second );
        buf.ends[buf.keys[x].second / 2][buf.keys[x].second % 2] = VertId( int( buf.defs.size() ) - 1 );
    }
    // every two segments among three points share an end, as two consecutive segments of a contour
    if ( buf.defs.size() <= 3 )
        return false;

    const auto [f, edges, verts] = cutFace( mesh.topology, segs[0].f, &preRes.vertFaces );
    if ( sortData )
    {
        auto& preds = buf.preds ? *buf.preds : buf.preds.emplace();
        const auto tri = preciseOwnTri( verts, mesh, *sortData );
        preds.reset( tri );
        for ( int p = 0; p < int( buf.defs.size() ); ++p )
        {
            const auto& s = segs[buf.defs[p] / 2];
            preds.addPoint( VertId( p ), preciseCutPoint( mesh, contours, s.c, s.i + buf.defs[p] % 2, edges, tri, *sortData ) );
        }
        return preds.failed() || PlanarTriangulation::hasCrossing( buf.ends, preds,
            buf.ends.size() > PlanarTriangulation::cMaxSegmentsInPairs ? buf.sweepCache() : nullptr );
    }

    const auto normal = cornersNormal( mesh, verts );
    if ( normal == Vector3f() )
        return true; // corners on one line give no plane to find the crossings in, so the face is bad, as in addMonotoneChords
    buf.xyz.resize( buf.defs.size() );
    for ( VertId p( 0 ); p < buf.xyz.size(); ++p )
    {
        const auto& s = segs[buf.defs[p] / 2];
        buf.xyz[p] = contours.intersection( s.c, s.i + buf.defs[p] % 2 ).coordinate;
    }
    return PlanarTriangulation::hasCrossing( buf.ends, buf.xyz, normal, buf.sweepCache() );
}

// finds the old faces of the mesh where two segments of the contours cross, right after doPreCutMesh; two segments with a common end,
// which follow each other along a contour or meet at a mesh vertex, do not cross
FaceBitSet findCrossingFaces( const Mesh& mesh, const OneMeshContours& contours, const PreCutResult& preRes, UndirectedEdgeId firstNewEdge,
    const SortIntersectionsData* sortData )
{
    MR_TIMER;
    // the new path edges (an edge of the old mesh is in no face)
    Vector<FaceSegment> segs;
    for ( int c = 0; c < preRes.paths.size(); ++c )
        for ( int i = 0; i < preRes.paths[c].size(); ++i )
            if ( const auto f = preRes.oldFaces( c )[i].left; f && preRes.paths[c][i].undirected() >= firstNewEdge )
                segs.push_back( { f, c, i } );
    tbb::parallel_sort( segs.begin(), segs.end(), [] ( const FaceSegment& a, const FaceSegment& b )
    {
        return std::tie( a.f, a.c, a.i ) < std::tie( b.f, b.c, b.i );
    } );

    FaceBitSet res( mesh.topology.faceSize() );
    tbb::enumerable_thread_specific<FaceCrossingsBuffers> buffers;
    ParallelFor( segs, buffers, [&] ( size_t i, FaceCrossingsBuffers& buf )
    {
        if ( i > 0 && segs[i - 1].f == segs[i].f )
            return; // not the first segment of its face
        size_t j = i + 1;
        while ( j < segs.size() && segs[j].f == segs[i].f )
            ++j;
        // isolate keeps the nested parallel loops of the sweep from stealing this thread onto another face with the same buffers
        if ( j > i + 1 && tbb::this_task_arena::isolate( [&] { return crossesInFace( buf, mesh, contours, preRes, sortData, { segs.data() + i, j - i } ); } ) )
            res.atomicTestSet( segs[i].f );
    } );
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
    Vector<float> sumLength; // only used if offset is variable
    struct InterInfo
    {
        float length{ 0.0f };
        int startPivot{ -1 };
        int endPivot{ -1 };
    };
    Vector<InterInfo> singleSegmentInfo; // only used if offset is variable
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
        if ( inter.isVert() )
            interVertId = inter.vert();
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

    // the offsets are filled in a vector of their own: GCC sees resize() of the default { 0 } as out of its bounds
    Vector<int> offsets( isoLines.size() + 1 );
    OneMeshContours res;
    for ( int i = 0; i < isoLines.size(); ++i )
    {
        // an iso-line that reaches the mesh boundary is open: it starts and ends on boundary edges
        assert( isoLines[i].size() > 1 );
        offsets[i + 1] = offsets[i] + int( isoLines[i].size() );
        if ( isoLines[i].front() == isoLines[i].back() )
            res.closed.autoResizeSet( i );
    }
    res.offsets = std::move( offsets );
    res.intersections.resizeNoInit( res.offsets.back() );
    parallelForContourPieces( res.offsets, [&] ( int i, int k0, int k1 )
    {
        for ( int k = k0; k < k1; ++k )
        {
            const auto& mep = isoLines[i][k - res.offsets[i]];
            auto& inter = res.intersections[k];

                inter.coordinate = meshAfterCut.edgePoint( mep );

                auto newUE = mep.e.undirected();
                if ( newUE < mesh.topology.undirectedEdgeSize() )
                    inter.setPrimitive( mep.e );
                else
                {
                    if ( nEM.splitEdges.test( newUE ) )
                    {
                        auto oldE = EdgeId( nEM.map[newUE] );
                        if ( mep.e.odd() )
                            oldE = oldE.sym();
                        inter.setPrimitive( oldE );
                    }
                    else
                    {
                        inter.setPrimitive( FaceId( nEM.map[newUE] ) );
                    }
                }
        }
    } );
    return res;
}

void reserveForCut( Mesh& mesh, const Mesh& sizeOf, const OneMeshContours& contours )
{
    size_t intersectedEdges = 0;
    for ( const auto& in : contours.intersections )
        if ( in.isEdge() )
            ++intersectedEdges;
    const size_t numVerts = contours.intersections.size() - contours.closed.count();
    const size_t numEdges = 2 * ( contours.intersections.size() + intersectedEdges );
    const auto& topology = sizeOf.topology;
    const auto totalExpectedVerts = topology.vertSize() + numVerts;

    // the fill of the holes adds about 0.5 times as many edges and 0.6 times as many faces as the cut adds edges: reserving for both here,
    // as many as the cut adds for headroom, copies the topology of the whole mesh to the larger buffers once, not again before the fill
    mesh.topology.vertReserve( totalExpectedVerts );
    mesh.topology.edgeReserve( topology.edgeSize() + 2 * numEdges );
    mesh.topology.faceReserve( topology.faceSize() + numEdges );
    mesh.points.reserve( totalExpectedVerts );
}

CutMeshResult cutMesh( Mesh& mesh, const OneMeshContours& contours, const CutMeshParameters& params )
{
    MR_TIMER;
    mesh.invalidateCaches();
    CutMeshResult res;
    if ( params.new2OldMap )
        prepareFacesMap( mesh.topology, *params.new2OldMap );

    const auto loneFaces = findLoneFaces( mesh.topology, contours );
    const auto firstNewEdge = UndirectedEdgeId( mesh.topology.undirectedEdgeSize() );
    auto preRes = doPreCutMesh( mesh, contours, params.sortData );
    // under force fill, the faces where the contours really cross (touching is fine) are bad too, found before the old edges are cut
    if ( params.forceFill )
        res.fbsWithContourIntersections = findCrossingFaces( mesh, contours, preRes, firstNewEdge, params.sortData );

    if ( params.new2oldEdgesMap )
    {
        Timer t( "new2oldEdgesMap" );
        for ( int i = 0; i < preRes.paths.size(); ++i )
        {
            for ( int j = 0; j < preRes.paths[i].size(); ++j )
            {
                params.new2oldEdgesMap->map[preRes.paths[i][j].undirected()] = preRes.oldFaces( i )[j].left;
            }
        }
    }

    cutEdgesIntoPieces( mesh, std::move( preRes.edgeData ), contours, params.sortData, params.new2OldMap, params.new2oldEdgesMap );
    FaceBitSet badLoneFaces;
    connectLoneContours( mesh, contours, loneFaces, preRes, firstNewEdge, params.sortData, params.new2oldEdgesMap, badLoneFaces );

    res.fbsWithContourIntersections = getBadFacesAfterCut( mesh.topology, preRes ) | badLoneFaces | res.fbsWithContourIntersections;
    if ( !params.forceFill && res.fbsWithContourIntersections.any() )
        return res;

    // find one edge for every hole to fill
    Timer t( "find edge per hole" );
    EdgeBitSet allHoleEdges( mesh.topology.edgeSize() );
    std::vector<EdgeId> holeRepresentativeEdges;
    Vector<FaceId> oldFaces; // of corresponding holeRepresentativeEdges
    Vector<EdgeId> pseudoHoleRepresentativeEdges;
    Vector<FaceId> pseudoOldFaces; // of corresponding pseudoHoleRepresentativeEdges
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
        const auto* oldEdgesInfo = preRes.oldFaces( pathId );
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

    Vector<Expected<OneMeshContour>> maybeOneMeshContours( contours.size() );
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

    OneMeshContours oneMeshContours;
    for ( auto& c : maybeOneMeshContours )
    {
        if ( !c )
            return unexpected( std::move( c.error() ) );
        oneMeshContours.push_back( *c );
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

    Vector<OneMeshContour> projected( contours.size() ); // every contour is found in parallel, then appended to cutConts
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
        projected[i] = std::move( *cutContRes );
    } );
    if ( ctx.is_group_execution_cancelled() )
        return unexpected( std::move( error ) );
    OneMeshContours cutConts;
    for ( const auto& c : projected )
        cutConts.push_back( c );

    auto cutRes = cutMesh( mesh, cutConts );
    if ( cutRes.fbsWithContourIntersections.any() )
        return unexpected( "Some projected contours have self-intersections" );
    return std::move( cutRes.resultCut );
}

} //namespace MR
