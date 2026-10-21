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
#include <numeric>

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

struct PathsEdgeIndex
{
    bool hasLeft{true};
    bool hasRight{true};
    FaceId right; // the face on the right of a path edge of the old mesh, while removedFaces has the one on its left
};

struct RemovedFaceInfo
{
    FaceId f;
    EdgeId leftRing[3];
};

using FullRemovedFacesInfo = std::vector<std::vector<RemovedFaceInfo>>;
using EdgeDataMap = ParallelHashMap<UndirectedEdgeId, EdgeData>;
struct PreCutResult
{
    EdgeDataMap edgeData;
    std::vector<EdgePath> paths;
    FullRemovedFacesInfo removedFaces;
    std::vector<std::vector<PathsEdgeIndex>> oldEdgesInfo;
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

// sets left face of given edge invalid and saves info about its old left ring
void invalidateFace( MeshTopology& topology, FullRemovedFacesInfo& removedFaceInfo, int contId, int interId, EdgeId leftEdge, size_t oldEdgesSize )
{
    auto thisFace = topology.left( leftEdge );
    if ( !thisFace.valid() )
        return;
    // fill removed tris
    auto& removedInfo = removedFaceInfo[contId][interId];
    removedInfo.f = thisFace;
    int leftEdgesCount = 0;
    for ( auto e : leftRing( topology, thisFace ) )
    {
        // we don't want to add new edges to this info structure
        // still they can be present in because of splices done before
        if ( e < oldEdgesSize )
        {
            if ( leftEdgesCount > 2 )
            {
                assert( false );
                break;
            }
            removedInfo.leftRing[leftEdgesCount] = e;
            ++leftEdgesCount;
        }
    }
    topology.setLeft( leftEdge, {} );
}

void iterateFindRemovedFaceInfo( FullRemovedFacesInfo& removedFaces, int contId, int interId, EdgeId leftEdge )
{
    for ( int backContId = contId; backContId >= 0; --backContId )
    {
        int prevInter = backContId == contId ? ( interId - 1 ) : ( int( removedFaces[backContId].size() ) - 1 );
        for ( int backInterId = prevInter; backInterId >= 0; --backInterId )
        {
            const auto& removedFace = removedFaces[backContId][backInterId];
            for ( auto e : removedFace.leftRing )
            {
                if ( e == leftEdge )
                {
                    removedFaces[contId][interId] = removedFace;
                    return;
                }
            }
        }
    }
}

// back-iterate removed face info to find correct splice edge for vert->face->vert like paths (second vert should find correct face edge for splice)
EdgeId iterateRemovedFacesInfoToFindLeftEdge( const MeshTopology& topology, const FullRemovedFacesInfo& removedFaces, int contId, int interId, FaceId f, VertId v )
{
    MR_TIMER;
    for ( int backContId = contId; backContId >= 0; --backContId )
    {
        int prevInter = backContId == contId ? ( interId - 1 ) : ( int( removedFaces[backContId].size() ) - 1 );
        for ( int backInterId = prevInter; backInterId >= 0; --backInterId )
        {
            const auto& removedFace = removedFaces[backContId][backInterId];
            if ( removedFace.f == f && removedFace.leftRing[0].valid() )
            {
                for ( auto e : orgRing( topology, v ) )
                {
                    if ( e == removedFace.leftRing[0] || e == removedFace.leftRing[1] || e == removedFace.leftRing[2] )
                    {
                        return e;
                    }
                }
            }
        }
    }
    return {};
}

PreCutResult doPreCutMesh( Mesh& mesh, const OneMeshContours& contours )
{
    MR_TIMER;

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
    const auto totalExpectedVerts = mesh.topology.vertSize() + numVerts;
    const auto totalExpectedEdges = mesh.topology.edgeSize() + numEdges;

    mesh.topology.vertReserve( totalExpectedVerts );
    mesh.topology.edgeReserve( totalExpectedEdges );

    PreCutResult res;
    res.paths.resize( contours.size() );
    res.oldEdgesInfo.resize( contours.size() );
    res.removedFaces.resize( contours.size() );
    res.edgeData.reserve( size_t( intersectedEdges ) );
    auto oldEdgesSize = mesh.topology.edgeSize();
    for ( int contourId = 0; contourId < contours.size(); ++contourId )
    {
        auto& removedFacesInfo = res.removedFaces[contourId];
        auto& oldEdgesInfo = res.oldEdgesInfo[contourId];
        auto& path = res.paths[contourId];
        const auto& inContour = contours[contourId].intersections;
        if ( inContour.size() < 2 )
            continue;
        bool closed = contours[contourId].closed;
        path.resize( inContour.size() - 1 );
        removedFacesInfo.resize( inContour.size() );
        oldEdgesInfo.resize( inContour.size() - 1 );
        VertId newVertId{};
        EdgeId newEdgeId{};
        for ( int intersectionId = 0; intersectionId < inContour.size(); ++intersectionId )
        {
            newVertId = {}; newEdgeId = {};

            const auto& inter = inContour[intersectionId];
            bool isVert = std::holds_alternative<VertId>( inter.primitiveId );
            bool isNextVert{false};
            if ( intersectionId + 1 < inContour.size() || !closed )
            {
                if ( isVert )
                    newVertId = std::get<VertId>( inter.primitiveId );
                else
                {
                    newVertId = mesh.topology.addVertId();
                    mesh.points.autoResizeAt( newVertId ) = inter.coordinate;
                }
            }
            // make edge (we don't need new edge if intersection is last one) and connect with this intersection
            if ( intersectionId + 1 < inContour.size() )
            {
                const auto& nextPrimId = inContour[intersectionId + 1].primitiveId;
                isNextVert = std::holds_alternative<VertId>( nextPrimId );

                if ( isVert )
                {
                    if ( isNextVert )
                    {
                        auto nextV = std::get<VertId>( nextPrimId );
                        for ( auto e : orgRing( mesh.topology, newVertId ) )
                        {
                            if ( mesh.topology.dest( e ) == nextV )
                            {
                                newEdgeId = e;
                                oldEdgesInfo[intersectionId].hasLeft = mesh.topology.left( newEdgeId ).valid();
                                oldEdgesInfo[intersectionId].hasRight = mesh.topology.right( newEdgeId ).valid();
                                oldEdgesInfo[intersectionId].right = mesh.topology.right( newEdgeId );
                                break;
                            }
                        }
                    }
                    else
                    {
                        newEdgeId = mesh.topology.makeEdge();
                        if ( std::holds_alternative<EdgeId>( nextPrimId ) )
                        {
                            auto e = std::get<EdgeId>( nextPrimId );
                            mesh.topology.splice( mesh.topology.next( e.sym() ).sym(), newEdgeId );
                            // the face right of e was already removed by the previous steps in it (e.g. along its edge to this vertex):
                            // take its info, otherwise the part on the other side of the new edge is not filled
                            if ( !mesh.topology.left( newEdgeId ) )
                                iterateFindRemovedFaceInfo( res.removedFaces, contourId, intersectionId, e.sym() );
                        }
                        else
                        {
                            auto f = std::get<FaceId>( nextPrimId );
                            EdgeId nextFaceIncidentLeft;
                            for ( auto e : orgRing( mesh.topology, newVertId ) )
                            {
                                if ( mesh.topology.left( e ) == f )
                                {
                                    nextFaceIncidentLeft = e;
                                    break;
                                }
                            }
                            if ( !nextFaceIncidentLeft )
                            {
                                // iterate backward to find removed face info
                                nextFaceIncidentLeft = iterateRemovedFacesInfoToFindLeftEdge( mesh.topology, res.removedFaces, contourId, intersectionId, f, newVertId );
                            }
                            assert( nextFaceIncidentLeft );
                            mesh.topology.splice( nextFaceIncidentLeft, newEdgeId );
                        }
                    }
                }
                else
                {
                    newEdgeId = mesh.topology.makeEdge();
                    mesh.topology.setOrg( newEdgeId, newVertId );
                }
            }

            if ( newEdgeId.valid() )
                path[intersectionId] = newEdgeId;
            // connect with prev
            if ( intersectionId > 0 )
            {
                if ( !isVert )
                {
                    if ( newEdgeId.valid() )
                        mesh.topology.splice( path[intersectionId - 1].sym(), newEdgeId );
                }
                else
                {
                    const auto& prevInterPrimId = inContour[intersectionId - 1].primitiveId;
                    if ( std::holds_alternative<EdgeId>( prevInterPrimId ) )
                    {
                        auto e = std::get<EdgeId>( prevInterPrimId );
                        invalidateFace( mesh.topology, res.removedFaces, contourId, intersectionId - 1, mesh.topology.next( e ).sym(), oldEdgesSize );
                        mesh.topology.splice( mesh.topology.next( e ).sym(), path[intersectionId - 1].sym() );
                    }
                    else if ( std::holds_alternative<FaceId>( prevInterPrimId ) )
                    {
                        auto currVert = newVertId;
                        if ( !currVert )
                            currVert = std::get<VertId>( inter.primitiveId ); // in case of start with vertex and closed
                        auto f = std::get<FaceId>( prevInterPrimId );
                        EdgeId prevFaceIncidentLeft;
                        for ( auto e : orgRing( mesh.topology, currVert ) )
                        {
                            if ( mesh.topology.left( e ) == f )
                            {
                                prevFaceIncidentLeft = e;
                                break;
                            }
                        }
                        // if prev face was removed need to find it other way
                        if ( !prevFaceIncidentLeft )
                        {
                            // iterate backward to find removed face info
                            prevFaceIncidentLeft = iterateRemovedFacesInfoToFindLeftEdge( mesh.topology, res.removedFaces, contourId, intersectionId, f, currVert );
                        }
                        assert( prevFaceIncidentLeft );
                        invalidateFace( mesh.topology, res.removedFaces, contourId, intersectionId - 1, prevFaceIncidentLeft, oldEdgesSize );
                        mesh.topology.splice( prevFaceIncidentLeft, path[intersectionId - 1].sym() );
                    }
                }
            }

            if ( newEdgeId.valid() )
            {
                invalidateFace( mesh.topology, res.removedFaces, contourId, intersectionId, newEdgeId, oldEdgesSize );
            }
            // add note to edgeData
            if ( newVertId.valid() && std::holds_alternative<EdgeId>( inter.primitiveId ) )
            {
                EdgeId thisEdge = std::get<EdgeId>( inter.primitiveId );
                auto& edgeData = res.edgeData[thisEdge.undirected()];
                edgeData.reserve( 5 ); // reseve small ammount to avoid overhead on reallocating for in most common scenarios
                edgeData.emplace_back( EdgeIntersectionData{
                    .interOnEdge = IntersectionData{ContourId( contourId ),IntersectionId( intersectionId )},
                    .newVert = newVertId,
                    .orgEdgeInLeftTri = newEdgeId,
                    .beforeSortIndex = int( edgeData.size() ) } );
                // fill removed tris
                if ( auto f = mesh.topology.left( thisEdge ) )
                    removedFacesInfo[intersectionId].f = f;
                else
                    iterateFindRemovedFaceInfo( res.removedFaces, contourId, intersectionId, thisEdge );
            }
            // fill removed tris
            if ( std::holds_alternative<FaceId>( inter.primitiveId ) )
                removedFacesInfo[intersectionId].f = std::get<FaceId>( inter.primitiveId );

        }
        if ( closed )
        {
            if ( !std::holds_alternative<VertId>( inContour.back().primitiveId ) )
                mesh.topology.splice( path.back().sym(), path.front() );
        }
        else
        {
            if ( !std::holds_alternative<VertId>( inContour.back().primitiveId ) )
            {
                assert( newVertId.valid() );
                mesh.topology.setOrg( path.back().sym(), newVertId );
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

/* this function triangulate holes where first and last edge are the same but sym
       / \
     /___  \
   /         \      hole starts with central edge going to triangle and finishes with it (shape can be any, not only triangle)
 /_____________\

edges should be already cut */
void fixOrphans( Mesh& mesh, const std::vector<EdgePath>& paths, const FullRemovedFacesInfo& removedFaces, FaceMap* new2OldMap, NewEdgesMap* new2OldEdgeMap )
{

    MR_TIMER;
    auto fixOrphan = [&]( EdgeId e, FaceId oldF )
    {
        if ( mesh.topology.left( e ).valid() ||
             mesh.topology.right( e ).valid() )
            return;

        auto next = mesh.topology.next( e.sym() );
        auto newEdge = mesh.topology.makeEdge();
        if ( new2OldEdgeMap )
            new2OldEdgeMap->map[newEdge.undirected()] = oldF;
        mesh.topology.splice( e, newEdge );
        mesh.topology.splice( next.sym(), newEdge.sym() );

        triangulateContour( mesh, e, oldF, new2OldMap, new2OldEdgeMap );
        triangulateContour( mesh, e.sym(), oldF, new2OldMap, new2OldEdgeMap );
    };
    for ( int i = 0; i < paths.size(); ++i )
    {
        const auto& path = paths[i];
        if(path.size() < 2)
            continue;

        FaceId oldF;
        // front
        auto e = path.front();
        if ( e == mesh.topology.next( e ) )
        {
            for ( int j = 0; j < path.size(); ++j )
            {
                oldF = removedFaces[i][j].f;
                if ( oldF.valid() )
                    break;
            }
            fixOrphan( e, oldF );
        }

        // back
        e = path.back().sym();
        if ( e == mesh.topology.next( e ) )
        {
            for ( int j = int( path.size() ) - 1; j >= 0; --j )
            {
                oldF = removedFaces[i][j].f;
                if ( oldF.valid() )
                    break;
            }
            fixOrphan( e, oldF );
        }
    }
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
            const auto f = preRes.removedFaces[i][j].f;
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

    auto ownVert = [&] ( VertId v ) { return preciseVert( v, mesh, !sortData->isOtherA, *sortData ); };
    // the exact definition of a vertex on the loops of a lone face
    auto pointOf = [&] ( const LoneFace& lf, VertId v )
    {
        PreciseCutPoint res;
        const auto it = vertInters.find( v );
        if ( it == vertInters.end() )
        {
            res.seg[0] = ownVert( v ); // a corner of the face
            return res;
        }
        const auto [c, i] = it->second;
        const auto& prim = contours[c].intersections[i].primitiveId;
        const auto& et = sortData->contours[c][i];
        const auto& otherTopology = sortData->otherMesh.topology;
        if ( const auto pe = std::get_if<EdgeId>( &prim ) )
        {
            // a side of the face crosses a triangle of the other mesh
            for ( int k = 0; k < 3; ++k )
                if ( lf.edges[k].undirected() == pe->undirected() )
                    res.seg = { ownVert( lf.verts[k] ), ownVert( lf.verts[( k + 1 ) % 3] ) };
            const auto t = otherTopology.getTriVerts( et.tri() );
            res.tri = { preciseOtherVert( t[0], *sortData ), preciseOtherVert( t[1], *sortData ), preciseOtherVert( t[2], *sortData ) };
        }
        else if ( std::holds_alternative<FaceId>( prim ) )
        {
            // an edge of the other mesh crosses the face
            res.seg = { preciseOtherVert( otherTopology.org( et.edge ), *sortData ), preciseOtherVert( otherTopology.dest( et.edge ), *sortData ) };
            res.tri = { ownVert( lf.verts[0] ), ownVert( lf.verts[1] ), ownVert( lf.verts[2] ) };
        }
        else
            res.seg[0] = ownVert( std::get<VertId>( prim ) );
        return res;
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
            plans[i] = PlanarTriangulation::getMonotonePlan( mesh, loops, { ownVert( lf.verts[0] ), ownVert( lf.verts[1] ), ownVert( lf.verts[2] ) },
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
        preciseVert( mesh.topology.org( intersectedEdge ), mesh, !sortData.isOtherA, sortData ),
        preciseVert( mesh.topology.dest( intersectedEdge ), mesh, !sortData.isOtherA, sortData ),
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
FaceBitSet getBadFacesAfterCut( const MeshTopology& topology, const PreCutResult& preRes,
                               const FullRemovedFacesInfo& oldFaces )
{
    MR_TIMER;
    FaceBitSet badFacesBS( topology.getValidFaces().size() );
    EdgeBitSet visited( topology.edgeSize() );
    for ( int pathId = 0; pathId < preRes.paths.size(); ++pathId )
    {
        const auto& path = preRes.paths[pathId];
        for ( int edgeId = 0; edgeId < path.size(); ++edgeId )
        {
            auto e0 = path[edgeId];
            if ( !preRes.removedFaces[pathId][edgeId].f.valid() )
                continue;

            if ( !preRes.oldEdgesInfo[pathId][edgeId].hasLeft &&
                 preRes.oldEdgesInfo[pathId][edgeId].hasRight )
                e0 = e0.sym();

            if ( visited.test( e0 ) || visited.test( e0.sym() ) )
                continue;
            for ( auto e : leftRing( topology, e0 ) )
            {
                visited.set( e );
                if ( e == e0.sym() )
                    badFacesBS.autoResizeSet( oldFaces[pathId][edgeId].f );
            }
        }
    }
    return badFacesBS;
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
    // .forceFillMode = CutMeshParameters::ForceFill::All
    // because we don't really care about intermediate mesh quality
    auto cutRes = cutMesh( meshAfterCut, initCutContoursVec, { .forceFillMode = CutMeshParameters::ForceFill::All,.new2oldEdgesMap = &nEM } );

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

    const auto loneFaces = findLoneFaces( mesh.topology, contours );
    const auto firstNewEdge = UndirectedEdgeId( mesh.topology.undirectedEdgeSize() );
    auto preRes = doPreCutMesh( mesh, contours );

    if ( params.new2oldEdgesMap )
    {
        Timer t( "new2oldEdgesMap" );
        for ( int i = 0; i < preRes.paths.size(); ++i )
        {
            for ( int j = 0; j < preRes.paths[i].size(); ++j )
            {
                params.new2oldEdgesMap->map[preRes.paths[i][j].undirected()] = preRes.removedFaces[i][j].f;
            }
        }
    }

    cutEdgesIntoPieces( mesh, std::move( preRes.edgeData ), contours, params.sortData, params.new2OldMap, params.new2oldEdgesMap );
    FaceBitSet badLoneFaces;
    connectLoneContours( mesh, contours, loneFaces, preRes, firstNewEdge, params.sortData, params.new2oldEdgesMap, badLoneFaces );
    fixOrphans( mesh, preRes.paths, preRes.removedFaces, params.new2OldMap, params.new2oldEdgesMap );

    res.fbsWithContourIntersections = getBadFacesAfterCut( mesh.topology, preRes, preRes.removedFaces ) | badLoneFaces;
    if ( params.forceFillMode == CutMeshParameters::ForceFill::None && res.fbsWithContourIntersections.any() )
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
            FaceId oldf = preRes.removedFaces[pathId][edgeId].f;
            if ( !oldf.valid() ||
                ( params.forceFillMode == CutMeshParameters::ForceFill::Good && res.fbsWithContourIntersections.test( oldf ) ) )
                continue;
            bool addLeftDesc = oldEdgesInfo[edgeId].hasLeft && !mesh.topology.left( path[edgeId] ).valid();
            bool addRightDesc = oldEdgesInfo[edgeId].hasRight && !mesh.topology.right( path[edgeId] ).valid();
            // the hole on the right of an edge of the old mesh is in its right face, which another contour or a lone one removed later
            const FaceId oldRightF = oldEdgesInfo[edgeId].right ? oldEdgesInfo[edgeId].right : oldf;
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
