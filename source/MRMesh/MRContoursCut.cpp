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
#include "MRTwoLineSegmDist.h"
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

using EdgeData = Vector<EdgeIntersectionData>;

// the faces of the old mesh on both sides of a path edge: the same face for a path edge inside it,
// the faces of an edge of the old mesh for a path edge along it
struct PathsEdgeIndex
{
    FaceId left;
    FaceId right;
};

using PathsOldFaces = Vector<Vector<PathsEdgeIndex>>; // [contour][edge]
using EdgeDataMap = ParallelHashMap<UndirectedEdgeId, EdgeData>;
struct PreCutResult
{
    EdgeDataMap edgeData;
    std::vector<EdgePath> paths;
    PathsOldFaces oldEdgesInfo;
    HashMap<FaceId, std::array<EdgeId, 3>> vertFaces; // the old faces of path edges at mesh vertices, removed by doPreCutMesh, and their edges
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
    const auto& prim = contours[c].intersections[i].primitiveId;
    const auto& et = sortData.contours[c][i];
    const auto& otherTopology = sortData.otherMesh.topology;
    if ( const auto pe = std::get_if<EdgeId>( &prim ) )
    {
        for ( int k = 0; k < 3; ++k )
            if ( edges[k].undirected() == pe->undirected() )
                res.seg = { tri[k], tri[( k + 1 ) % 3] };
        res.tri = preciseOtherTri( et.tri(), sortData );
    }
    else if ( std::holds_alternative<FaceId>( prim ) )
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
        res.seg[0] = preciseOwnVert( std::get<VertId>( prim ), mesh, sortData );
    return res;
}

PreCutResult doPreCutMesh( Mesh& mesh, const OneMeshContours& contours, const SortIntersectionsData* sortData )
{
    MR_TIMER;
    auto& topology = mesh.topology;

    int intersectedEdges = 0;
    for ( const auto& cont : contours )
        for ( const auto& in : cont.intersections )
            if ( std::holds_alternative<EdgeId>( in.primitiveId ) )
                ++intersectedEdges;
    reserveForCut( mesh, mesh, contours );

    PreCutResult res;
    res.paths.resize( contours.size() );
    res.oldEdgesInfo.resize( contours.size() );
    res.edgeData.reserve( size_t( intersectedEdges ) );
    const auto oldEdgesSize = topology.edgeSize();

    // find the old face of every path edge before the topology changes;
    // the faces of path edges at mesh vertices are removed here (the others by cutOneEdge or connectLoneContours),
    // and their edges show where to splice the path edges around the vertices
    auto& vertFaces = res.vertFaces;
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
            else if ( const auto pe = std::get_if<EdgeId>( va ? &b : &a ) )
            {
                // a path edge leaves an edge point into the left face of the edge and arrives at it from the right face
                const auto e = va ? pe->sym() : *pe;
                const auto v = va ? *va : vb ? *vb : VertId{};
                // precondition: a mesh vertex at the other end is the third vertex of that face, so the contour does not fold back
                // at the edge; a fold puts the path edge in the other face, where Release still gets a valid topology
                // (a mesh vertex that is no corner of these faces is not handled)
                const bool fold = v && ( !topology.left( e ) || topology.dest( topology.next( e ) ) != v );
                assert( !fold );
                faces[i].left = faces[i].right = topology.left( fold ? e.sym() : e );
            }
            else
                faces[i].left = faces[i].right = std::get<FaceId>( va ? b : a );
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

// a face of the mesh that contours go inside, with its sides before the cut
struct CutFace
{
    FaceId f;
    std::array<EdgeId, 3> edges; // counterclockwise, before the cut
    ThreeVertIds verts; // verts[i] = org( edges[i] )
};

// face f with its sides before the cut changes their topology (the faces that doPreCutMesh removed have their edges in vertFaces)
CutFace cutFace( const MeshTopology& topology, FaceId f, const HashMap<FaceId, std::array<EdgeId, 3>>& vertFaces = {} )
{
    CutFace res{ f };
    if ( const auto it = vertFaces.find( f ); it != vertFaces.end() )
        res.edges = it->second;
    else
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
            [] ( const OneMeshIntersection& i ) { return std::holds_alternative<FaceId>( i.primitiveId ); } ) )
            faces.push_back( std::get<FaceId>( c.intersections.front().primitiveId ) );
    }
    std::sort( faces.begin(), faces.end() );
    faces.erase( std::unique( faces.begin(), faces.end() ), faces.end() );
    Vector<CutFace> res;
    for ( auto f : faces )
        res.push_back( cutFace( topology, f ) );
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

// the index of the face in sorted faces, or -1
int findCutFace( const Vector<CutFace>& faces, FaceId f )
{
    const auto it = std::lower_bound( faces.begin(), faces.end(), f, [] ( const CutFace& cf, FaceId f ) { return cf.f < f; } );
    return it == faces.end() || it->f != f ? -1 : int( it - faces.begin() );
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
    // the definition of every vertex at a split crossing of two paths (see splitContourCrossings)
    HashMap<VertId, PreciseCutPoint> crossings;

    std::array<PreciseVertCoords, 3> triangle( const CutFace& cf ) const { return preciseOwnTri( cf.verts, mesh, *sortData ); }

    // the definition of vertex v on the loops of face cf with precise corners tri
    PreciseCutPoint point( const CutFace& cf, const std::array<PreciseVertCoords, 3>& tri, VertId v ) const
    {
        if ( const auto it = crossings.find( v ); it != crossings.end() )
            return it->second;
        const auto it = vertInters.find( v );
        if ( it == vertInters.end() )
        {
            PreciseCutPoint res;
            res.seg[0] = preciseOwnVert( v, mesh, *sortData ); // a corner of the face
            return res;
        }
        return preciseCutPoint( mesh, contours, it->second.first, it->second.second, cf.edges, tri, *sortData );
    }

    // whether v can be a vertex on the loops of face cf: a corner of it, a point on its side, or a point inside it
    bool isPointOf( const CutFace& cf, VertId v ) const
    {
        const auto isCorner = [&] ( VertId u ) { return u == cf.verts[0] || u == cf.verts[1] || u == cf.verts[2]; };
        const auto it = vertInters.find( v );
        if ( it == vertInters.end() )
            return isCorner( v );
        const auto& prim = contours[it->second.first].intersections[it->second.second].primitiveId;
        if ( const auto pe = std::get_if<EdgeId>( &prim ) )
            return std::any_of( cf.edges.begin(), cf.edges.end(), [&] ( EdgeId s ) { return s.undirected() == pe->undirected(); } );
        if ( const auto f = std::get_if<FaceId>( &prim ) )
            return *f == cf.f;
        return isCorner( std::get<VertId>( prim ) );
    }

    // the triangle of the other mesh along whose plane contour c goes inside a face from its point j to point j + 1, invalid if unknown:
    // the one common to both points, of the triangle that a side of the face crosses at a point or the two of the edge crossing the face there
    FaceId cutTriangle( int c, int j ) const
    {
        const auto& otherTopology = sortData->otherMesh.topology;
        auto trianglesAt = [&] ( int k ) -> std::array<FaceId, 2>
        {
            const auto& prim = contours[c].intersections[k].primitiveId;
            const auto& et = sortData->contours[c][k];
            if ( std::holds_alternative<EdgeId>( prim ) )
                return { et.tri(), FaceId{} };
            if ( std::holds_alternative<FaceId>( prim ) )
                return { otherTopology.left( et.edge ), otherTopology.right( et.edge ) };
            return {}; // a vertex of the face
        };
        const auto a = trianglesAt( j ), b = trianglesAt( j + 1 );
        for ( auto t : a )
            if ( t && ( t == b[0] || t == b[1] ) )
                return t;
        return {};
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
    BitSet removedHere; // the faces that no other contour removed, to put back if nothing changes in them
};

// removes the faces that no other contour removed, and finds the holes of all of them; sets the contour points at the ends of the path edges
// in exact if it has sortData, and the cut triangle of every path edge in edgeCuts (see ExactCutPoints::cutTriangle) if it is given
FaceHoles makeFaceHoles( MeshTopology& topology, const Vector<CutFace>& faces, const PreCutResult& preRes, UndirectedEdgeId firstNewEdge,
    ExactCutPoints& exact, HashMap<UndirectedEdgeId, FaceId>* edgeCuts = nullptr )
{
    FaceHoles res{ Vector<EdgePath>( faces.size() ), Vector<EdgeLoops>( faces.size() ), BitSet( faces.size() ) };
    for ( int i = 0; i < preRes.paths.size(); ++i )
    {
        for ( int j = 0; j < preRes.paths[i].size(); ++j )
        {
            const int k = findCutFace( faces, preRes.oldEdgesInfo[i][j].left );
            if ( k < 0 )
                continue;
            auto& edges = res.edges[k];
            const auto e = preRes.paths[i][j];
            edges.push_back( e );
            if ( e.undirected() >= firstNewEdge ) // a new edge is inside the face, while an old one has it on the left only
                edges.push_back( e.sym() );
            if ( !exact.sortData )
                continue;
            exact.vertInters[topology.org( e )] = { i, j };
            exact.vertInters[topology.dest( e )] = { i, j + 1 };
            if ( edgeCuts )
                ( *edgeCuts )[e.undirected()] = exact.cutTriangle( i, j );
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
// consistent with the boolean if exact has sortData; the faces where that fails are added to badFaces, and put back if removed here;
// faces without loops are skipped
void addMonotoneChords( Mesh& mesh, const Vector<CutFace>& faces, const Vector<EdgeLoops>& faceLoops, const BitSet& removedHere,
    const ExactCutPoints& exact, NewEdgesMap* new2OldEdgeMap, FaceBitSet& badFaces )
{
    // the faces are independent, so plan them in parallel
    Vector<std::optional<HoleFillPlan>> plans( faces.size() );
    ParallelFor( plans, [&] ( size_t i )
    {
        const auto& cf = faces[i];
        const auto& loops = faceLoops[i];
        if ( loops.empty() )
            return;
        if ( exact.sortData )
        {
            const auto tri = exact.triangle( cf );
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
        if ( faceLoops[i].empty() )
            continue;
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

// splits both edges of every crossing at a new vertex, in the order that findCrossings gives (along every edge), as MeshTopology::splitEdge does;
// adds the pieces of every split edge to `pieces` in the order along the edge, the last one keeping the id of the edge; returns the first new vertex
VertId splitCrossings( Mesh& mesh, const Vector<EdgePair>& crossings, FaceId oldFace, HashMap<UndirectedEdgeId, EdgePath>& pieces,
    NewEdgesMap* new2OldEdgeMap )
{
    auto& tp = mesh.topology;
    const VertId firstNewVert( tp.vertSize() );
    // in the middle of the closest points of the edges, before the edges change
    for ( const auto& [a, b] : crossings )
    {
        const auto cp = findTwoLineSegmClosestPoints( LineSegm3d( mesh.edgeSegment( a ) ), LineSegm3d( mesh.edgeSegment( b ) ) );
        mesh.addPoint( Vector3f( 0.5 * ( cp.a + cp.b ) ) );
    }

    for ( int i = 0; i < crossings.size(); ++i )
    {
        // around the new vertex counterclockwise: the upper edge to its destination, the lower edge to its destination,
        // the upper edge to its origin, the lower edge to its origin
        std::array<EdgeId, 4> ring{ crossings[i].second, crossings[i].first };
        for ( int j = 0; j < 2; ++j )
        {
            const EdgeId e = ring[j];
            assert( !tp.left( e ) && !tp.right( e ) );
            const EdgeId piece = tp.makeEdge();
            if ( new2OldEdgeMap )
                new2OldEdgeMap->map[piece.undirected()] = oldFace;
            // the piece goes after the edge at its origin, and then the edge leaves it
            tp.splice( e, piece );
            tp.splice( tp.prev( e ), e );
            ring[j + 2] = piece.sym();
            auto& ps = pieces[e.undirected()];
            if ( ps.empty() )
                ps.push_back( e ); // the last piece
            ps.insert( ps.end() - 1, piece );
        }
        tp.splice( ring[0], ring[1] );
        tp.splice( ring[1], ring[2] );
        tp.splice( ring[2], ring[3] );
        tp.setOrg( ring[0], firstNewVert + i );
    }
    return firstNewVert;
}

// splits the path edges inside the given faces (where contours cross, see findCrossingFaces) at new vertices shared by the crossing paths,
// so that the holes of these faces can be filled as usual, and splits these holes in monotone parts (see addMonotoneChords), which connects
// the islands of lone contours left in them with the rest of their holes; a face where an edge of the paths crosses a side or the predicates
// fail stays as it is, and is added to unsplitFaces; returns for every split edge its pieces (see splitCrossings)
HashMap<UndirectedEdgeId, EdgePath> splitContourCrossings( Mesh& mesh, const OneMeshContours& contours, const Vector<CutFace>& faces,
    FaceBitSet& badFaces, FaceBitSet& unsplitFaces, const PreCutResult& preRes, UndirectedEdgeId firstNewEdge,
    const SortIntersectionsData& sortData, NewEdgesMap* new2OldEdgeMap )
{
    MR_TIMER;
    HashMap<UndirectedEdgeId, EdgePath> pieces;

    // the holes of every face, also of the ones put back after the plans of their lone contours failed (see connectLoneContours)
    ExactCutPoints exact{ mesh, contours, &sortData };
    HashMap<UndirectedEdgeId, FaceId> edgeCuts;
    auto holes = makeFaceHoles( mesh.topology, faces, preRes, firstNewEdge, exact, &edgeCuts );
    auto& tp = mesh.topology;
    auto cutOf = [&] ( UndirectedEdgeId ue )
    {
        const auto t = getAt( edgeCuts, ue );
        return t ? preciseOtherTri( t, sortData ) : std::array<PreciseVertCoords, 3>{}; // invalid ids for an edge along a side
    };

    // the faces are independent, so find their crossings in parallel
    Vector<std::optional<Vector<EdgePair>>> crossings( faces.size() );
    ParallelFor( crossings, [&] ( size_t k )
    {
        const auto& cf = faces[k];
        for ( const auto& loop : holes.loops[k] )
            for ( auto e : loop )
                if ( !exact.isPointOf( cf, tp.org( e ) ) )
                    return;
        const auto tri = exact.triangle( cf );
        crossings[k] = PlanarTriangulation::findCrossings( mesh, holes.loops[k], tri, [&] ( VertId v ) { return exact.point( cf, tri, v ); }, cutOf );
    } );

    // split them face by face
    for ( int k = 0; k < faces.size(); ++k )
    {
        const auto& cf = faces[k];
        if ( !crossings[k] || crossings[k]->empty() )
        {
            unsplitFaces.autoResizeSet( cf.f );
            if ( holes.removedHere.test( k ) )
                tp.setLeft( cf.edges[0], cf.f );
            continue;
        }
        auto newVert = splitCrossings( mesh, *crossings[k], cf.f, pieces, new2OldEdgeMap );
        for ( const auto& [a, b] : *crossings[k] )
        {
            exact.crossings[newVert++] = { .tri = cutOf( a.undirected() ), .tri2 = cutOf( b.undirected() ) };
            for ( auto e : { a, b } )
                for ( auto p : pieces.at( e.undirected() ) )
                    holes.edges[k].insert( holes.edges[k].end(), { p, p.sym() } );
        }
    }

    // the chords on the loops with the pieces connect the islands left in the split faces with the rest of their holes; and the monotone
    // parts keep the fill from folding where the coordinates of the points disagree with the predicates (as at a point nearly on another
    // segment); where that fails, the holes are filled as they are, and the islands of lone contours stay unconnected, filled on both sides
    // (see connectLoneContours); these faces are in badFaces already
    EdgeBitSet inLoops( tp.edgeSize() );
    for ( int k = 0; k < faces.size(); ++k )
        holes.loops[k] = unsplitFaces.test( faces[k].f ) ? EdgeLoops{} : holeLoops( tp, holes.edges[k], inLoops );
    addMonotoneChords( mesh, faces, holes.loops, holes.removedHere, exact, new2OldEdgeMap, badFaces );
    return pieces;
}

// replaces every split edge of the paths by its pieces, which take the old faces of the edge
void insertPieces( PreCutResult& preRes, const HashMap<UndirectedEdgeId, EdgePath>& pieces )
{
    if ( pieces.empty() )
        return;
    for ( int i = 0; i < preRes.paths.size(); ++i )
    {
        auto& path = preRes.paths[i];
        if ( std::none_of( path.begin(), path.end(), [&] ( EdgeId e ) { return pieces.contains( e.undirected() ); } ) )
            continue;
        auto& oldEdgesInfo = preRes.oldEdgesInfo[i];
        EdgePath newPath;
        Vector<PathsEdgeIndex> newOldEdgesInfo;
        for ( int j = 0; j < path.size(); ++j )
        {
            const auto e = path[j];
            const auto it = pieces.find( e.undirected() );
            if ( it == pieces.end() )
                newPath.push_back( e );
            else if ( it->second.back() == e ) // the pieces go along the last one, which keeps the id of the edge
                newPath.insert( newPath.end(), it->second.begin(), it->second.end() );
            else
                for ( auto p = it->second.rbegin(); p != it->second.rend(); ++p )
                    newPath.push_back( p->sym() );
            newOldEdgesInfo.resize( newPath.size(), oldEdgesInfo[j] );
        }
        path = std::move( newPath );
        oldEdgesInfo = std::move( newOldEdgesInfo );
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

    const auto [f, edges, verts] = cutFace( mesh.topology, segs[0].f, preRes.vertFaces );
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
        return true; // corners on one line give no plane to find the crossings in, so the face is bad, as in connectLoneContours
    buf.xyz.resize( buf.defs.size() );
    for ( VertId p( 0 ); p < buf.xyz.size(); ++p )
    {
        const auto& s = segs[buf.defs[p] / 2];
        buf.xyz[p] = contours[s.c].intersections[s.i + buf.defs[p] % 2].coordinate;
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
            if ( const auto f = preRes.oldEdgesInfo[c][i].left; f && preRes.paths[c][i].undirected() >= firstNewEdge )
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

void reserveForCut( Mesh& mesh, const Mesh& sizeOf, const OneMeshContours& contours )
{
    size_t numVerts = 0;
    size_t numEdges = 0;
    size_t intersectedEdges = 0;
    for ( const auto& cont : contours )
    {
        const auto size = cont.intersections.size();
        numVerts += size;
        numEdges += 2 * size;
        for ( const auto& in : cont.intersections )
            if ( std::holds_alternative<EdgeId>( in.primitiveId ) )
                ++intersectedEdges;
        if ( cont.closed )
            --numVerts;
    }
    numEdges += 2 * intersectedEdges;
    const auto& topology = sizeOf.topology;
    const auto totalExpectedVerts = topology.vertSize() + numVerts;

    // the fill of the holes adds about 0.5 times as many edges and 0.6 times as many faces as the cut adds edges: reserving for both here,
    // as many as the cut adds for headroom, copies the topology of the whole mesh to the larger buffers once, not again before the fill
    mesh.topology.vertReserve( totalExpectedVerts );
    mesh.topology.edgeReserve( topology.edgeSize() + 2 * numEdges );
    mesh.topology.faceReserve( topology.faceSize() + numEdges );
    mesh.points.reserve( totalExpectedVerts );
}

// cutMesh that also gives in unsplitCrossings the triangles of the cut mesh in the bad faces where the crossing contours could not be split
// (as at the crossings with both sides of a double-sided triangle, which the exact predicates cannot tell apart), where the fill can fold;
// the boolean keeps or removes them one by one (see doBooleanOperationImpl)
CutMeshResult cutMeshImpl( Mesh& mesh, const OneMeshContours& contours, const CutMeshParameters& params, FaceBitSet* unsplitCrossings )
{
    MR_TIMER;
    mesh.invalidateCaches();
    CutMeshResult res;
    if ( params.new2OldMap )
        prepareFacesMap( mesh.topology, *params.new2OldMap );

    const auto loneFaces = findLoneFaces( mesh.topology, contours );
    const auto firstNewEdge = UndirectedEdgeId( mesh.topology.undirectedEdgeSize() );
    auto preRes = doPreCutMesh( mesh, contours, params.sortData );
    // under force fill, the faces where the contours really cross (touching is fine) are bad too, found before the old edges are cut;
    // with sortData, the contours are split where they cross in these faces, which needs their sides before the cut
    Vector<CutFace> crossingFaces;
    if ( params.forceFill )
    {
        res.fbsWithContourIntersections = findCrossingFaces( mesh, contours, preRes, firstNewEdge, params.sortData );
        if ( params.sortData )
        {
            crossingFaces.reserve( res.fbsWithContourIntersections.count() );
            for ( auto f : res.fbsWithContourIntersections )
                crossingFaces.push_back( cutFace( mesh.topology, f, preRes.vertFaces ) );
        }
    }

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

    res.fbsWithContourIntersections = getBadFacesAfterCut( mesh.topology, preRes ) | badLoneFaces | res.fbsWithContourIntersections;
    if ( !params.forceFill && res.fbsWithContourIntersections.any() )
        return res;
    // the crossing contours are split at the crossings, so that the holes of their faces can be filled as usual; the faces stay reported
    FaceBitSet unsplitFaces; // the faces where the crossings stay unsplit
    FaceBitSet unused;
    auto& outUnsplit = unsplitCrossings ? *unsplitCrossings : unused;
    if ( !crossingFaces.empty() )
        insertPieces( preRes, splitContourCrossings( mesh, contours, crossingFaces, res.fbsWithContourIntersections, unsplitFaces,
            preRes, firstNewEdge, *params.sortData, params.new2oldEdgesMap ) );

    // find one edge for every hole to fill
    Timer t( "find edge per hole" );
    EdgeBitSet allHoleEdges( mesh.topology.edgeSize() );
    std::vector<EdgeId> holeRepresentativeEdges;
    Vector<FaceId> oldFaces; // of corresponding holeRepresentativeEdges
    Vector<EdgeId> pseudoHoleRepresentativeEdges;
    Vector<FaceId> pseudoOldFaces; // of corresponding pseudoHoleRepresentativeEdges
    const bool needOldFaces = params.new2OldMap || params.new2oldEdgesMap || unsplitFaces.any();
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

        const auto firstFace = mesh.topology.faceSize();
        executeTriangulateContourPlan( mesh, holeRepresentativeEdges[i], fillPlans[i],
            needOldFaces ? oldFaces[i] : FaceId{}, params.new2OldMap, params.new2oldEdgesMap );
        if ( needOldFaces && unsplitFaces.test( oldFaces[i] ) )
            outUnsplit.autoResizeSet( FaceId( firstFace ), mesh.topology.faceSize() - firstFace );
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
        if ( needOldFaces && unsplitFaces.test( pseudoOldFaces[i] ) )
            outUnsplit.autoResizeSet( mesh.topology.lastValidFace() );
    }
    // the faces put back whole, with the islands of their lone contours filled on both sides (see connectLoneContours)
    for ( auto f : unsplitFaces )
        if ( mesh.topology.hasFace( f ) )
            outUnsplit.autoResizeSet( f );

    assert( mesh.topology.faceSize() == expectedTotalTris );
    if ( params.new2OldMap )
        assert( params.new2OldMap->size() == ( numTris != 0 ? expectedTotalTris : mesh.topology.lastValidFace() + 1 ) );

    res.resultCut = std::move( preRes.paths );

    return res;
}

CutMeshResult cutMesh( Mesh& mesh, const OneMeshContours& contours, const CutMeshParameters& params )
{
    return cutMeshImpl( mesh, contours, params, nullptr );
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

    Vector<OneMeshContour> oneMeshContours( contours.size() );
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
