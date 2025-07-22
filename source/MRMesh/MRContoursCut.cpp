#include "MRContoursCut.h"
#include "MRMesh.h"
#include "MRPolyline.h"
#include "MRTriangleIntersection.h"
#include "MRMeshTopology.h"
#include "MRMeshDelone.h"
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
#include "MRPch/MRSpdlog.h"
#include <parallel_hashmap/phmap.h>
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

enum class TrianglesSortRes
{
    Undetermined, // triangles positions cannot be determined
    Left,         // second triangle is form left side of oriented ABC
    Right         // second triangle is form right side of oriented ABC
};

using EdgeData = std::vector<EdgeIntersectionData>;

struct PathsEdgeIndex
{
    bool hasLeft{true};
    bool hasRight{true};
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

// Indicates if one of sorted faces was reverted in contour (only can be during propagation sort)
enum class EdgeSortState
{
    Straight, // both intersection edges are in original state
    LReverted, // left sort candidate returned
    RReverted // right sort candidate returned
};

void preparePreciseVerts( const SortIntersectionsData& sortData, const VertId* verts, PreciseVertCoords* preciseVerts, int n )
{
    if ( sortData.isOtherA )
    {
        for ( int i = 0; i < n; ++i )
            preciseVerts[i] = {verts[i],sortData.converter( sortData.otherMesh.points[verts[i]] )};
    }
    else
    {
        if ( !sortData.rigidB2A )
        {
            for ( int i = 0; i < n; ++i )
                preciseVerts[i] = {verts[i] + int( sortData.meshAVertsNum ),sortData.converter( sortData.otherMesh.points[verts[i]] )};
        }
        else
        {
            for ( int i = 0; i < n; ++i )
                preciseVerts[i] = {verts[i] + int( sortData.meshAVertsNum ),sortData.converter( ( *sortData.rigidB2A )( sortData.otherMesh.points[verts[i]] ) )};
        }
    }
}

TrianglesSortRes sortTrianglesSharedEdge( const SortIntersectionsData& sortData, EdgeId  sharedEdge )
{
    const auto& topology = sortData.otherMesh.topology;

    std::array<PreciseVertCoords, 4> preciseVerts;
    std::array<VertId, 4> verts;
    verts[0] = topology.dest( topology.next( sharedEdge ) );
    verts[1] = topology.org( sharedEdge );
    verts[2] = topology.dest( sharedEdge );
    verts[3] = topology.dest( topology.prev( sharedEdge ) );

    if ( verts[0] == verts[3] )
        return TrianglesSortRes::Undetermined;

    preparePreciseVerts( sortData, verts.data(), preciseVerts.data(), 4 );

    if ( orient3d( preciseVerts ) )
        return TrianglesSortRes::Left;
    else
        return TrianglesSortRes::Right;
}

TrianglesSortRes sortTrianglesSharedVert( const SortIntersectionsData& sortData, FaceId fl, EdgeId sharedVertOrg )
{
    const auto& topology = sortData.otherMesh.topology;
    const auto& edgePerFaces = topology.edgePerFace();
    auto el = edgePerFaces[fl];

    std::array<PreciseVertCoords, 5> preciseVerts;
    std::array<VertId, 5> verts;
    verts[0] = topology.org( el );
    verts[1] = topology.dest( el );
    verts[2] = topology.dest( topology.next( el ) );
    verts[3] = topology.dest( sharedVertOrg );
    verts[4] = topology.dest( topology.next( sharedVertOrg ) );

    // check multiple case
    bool multiple3 = verts[3] == verts[0] || verts[3] == verts[1] || verts[3] == verts[2];
    bool multiple4 = verts[4] == verts[0] || verts[4] == verts[1] || verts[4] == verts[2];
    if ( multiple3 && multiple4 )
        return TrianglesSortRes::Undetermined;
    if ( multiple3 )
        std::swap( preciseVerts[3], preciseVerts[4] );
    if ( multiple3 || multiple4 )
    {
        preparePreciseVerts( sortData, verts.data(), preciseVerts.data(), 4 );
        if ( orient3d( preciseVerts.data() ) )
            return TrianglesSortRes::Left;
        else
            return TrianglesSortRes::Right;
    }

    // common non-multiple case
    preparePreciseVerts( sortData, verts.data(), preciseVerts.data(), 5 );

    bool oneRes = orient3d( preciseVerts.data() );
    std::swap( preciseVerts[3], preciseVerts[4] );
    bool otherRes = orient3d( preciseVerts.data() );

    if ( oneRes != otherRes )
        return TrianglesSortRes::Undetermined;
    else if ( oneRes )
        return TrianglesSortRes::Left;
    else
        return TrianglesSortRes::Right;
}

TrianglesSortRes sortTrianglesNoShared( const SortIntersectionsData& sortData, FaceId fl, FaceId fr )
{
    const auto& topology = sortData.otherMesh.topology;
    const auto& edgePerFaces = topology.edgePerFace();
    auto el = edgePerFaces[fl];
    auto er = edgePerFaces[fr];

    std::array<PreciseVertCoords, 6> preciseVerts;
    std::array<VertId, 6> verts;
    verts[0] = topology.org( el );
    verts[1] = topology.dest( el );
    verts[2] = topology.dest( topology.next( el ) );
    verts[3] = topology.org( er );
    verts[4] = topology.dest( er );
    verts[5] = topology.dest( topology.next( er ) );

    preparePreciseVerts( sortData, verts.data(), preciseVerts.data(), 6 );

    bool arRes = orient3d( preciseVerts.data() );
    std::swap( preciseVerts[3], preciseVerts[4] );
    bool brRes = orient3d( preciseVerts.data() );
    std::swap( preciseVerts[3], preciseVerts[5] );
    bool crRes = orient3d( preciseVerts.data() );

    if ( arRes != brRes || arRes != crRes )
        return TrianglesSortRes::Undetermined;
    else if ( arRes )
        return TrianglesSortRes::Left;
    else
        return TrianglesSortRes::Right;
}

TrianglesSortRes sortTriangles( const SortIntersectionsData& sortData, FaceId fl, FaceId fr )
{
    const auto& topology = sortData.otherMesh.topology;
    EdgeId sharedEdge = topology.sharedEdge( fl, fr );
    if ( sharedEdge.valid() )
        return sortTrianglesSharedEdge( sortData, sharedEdge );

    sharedEdge = topology.sharedVertInOrg( fl, fr );
    if ( sharedEdge.valid() )
        return sortTrianglesSharedVert( sortData, fl, sharedEdge );

    return sortTrianglesNoShared( sortData, fl, fr );
}

// Try sort left face by right, and right by left
TrianglesSortRes sortTrianglesSymmetrical( const SortIntersectionsData& sortData,
    EdgeId el, EdgeId er,
    FaceId fl, FaceId fr, EdgeId baseEdgeOr, EdgeSortState state )
{
    // try sort right face by left
    TrianglesSortRes res = sortTriangles( sortData, fl, fr );
    if ( res != TrianglesSortRes::Undetermined )
    {
        bool correctOrder = ( state == EdgeSortState::LReverted ) ? ( el != baseEdgeOr ) : ( el == baseEdgeOr );
        return correctOrder == ( res == TrianglesSortRes::Left ) ?
            TrianglesSortRes::Left : TrianglesSortRes::Right;
    }
    // try sort left face by right
    res = sortTriangles( sortData, fr, fl );
    if ( res != TrianglesSortRes::Undetermined )
    {
        bool correctOrder = ( state == EdgeSortState::RReverted ) ? ( er != baseEdgeOr ) : ( er == baseEdgeOr );
        return correctOrder == ( res == TrianglesSortRes::Right ) ?
            TrianglesSortRes::Left : TrianglesSortRes::Right;
    }
    return TrianglesSortRes::Undetermined;
}

// try determine sort looking on next or prev intersection
TrianglesSortRes sortPropagateContour(
    const MeshTopology& tp,
    const SortIntersectionsData& sortData,
    const IntersectionData& il, const IntersectionData& ir,
    EdgeId baseEdgeOr )
{
    const auto& lContour = sortData.contours[il.contourId];
    const auto& rContour = sortData.contours[ir.contourId];
    const EdgeId el = lContour[il.intersectionId].edge;
    const EdgeId er = rContour[ir.intersectionId].edge;

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
    EdgeId lastCommonEdgeNext = baseEdgeOr;
    EdgeId lastCommonEdgePrev = baseEdgeOr;
    // check if next/prev intersection can determine sort
    auto checkOther = [&] ( bool next )->TrianglesSortRes
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
            return TrianglesSortRes::Undetermined;
        }
        auto otherR = getNextPrev( rOtherRef, sameContour ? lOtherRef : rOtherRef, false, next );
        if ( !otherR )
        {
            tryThis = false; // terminal (end of contour reached
            return TrianglesSortRes::Undetermined;
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
                return TrianglesSortRes::Undetermined;

            FaceId fl;
            FaceId fr;
            EdgeSortState state;
            if ( lReturned )
            {
                fl = lContour[lOtherRef].tri();
                fr = rContour[startR].tri();
                state = EdgeSortState::LReverted;
            }
            else
            {
                assert( rReturned );
                fl = lContour[startL].tri();
                fr = rContour[rOtherRef].tri();
                state = EdgeSortState::RReverted;
            }
            return sortTrianglesSymmetrical( sortData, el, er, fl, fr, baseEdgeOr, state );
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
                return sortData.isOtherA ? TrianglesSortRes::Left : TrianglesSortRes::Right; // terminal
            else if ( otherER == tp.next( lastCommonEdgeRef ).undirected() || otherER == tp.prev( lastCommonEdgeRef ).undirected() )
                return sortData.isOtherA ? TrianglesSortRes::Right : TrianglesSortRes::Left; // terminal
            else
            {
                // TODO: support this case
                // we can be here only if doing self-boolean of non-closed contour passing through vertex
                tryThis = false; // for now just terminate, for simplicity
                return TrianglesSortRes::Undetermined;
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
            return TrianglesSortRes::Undetermined; // go next if we came to same intersection 

        return sortTrianglesSymmetrical( sortData, el, er, fl, fr, baseEdgeOr, EdgeSortState::Straight );
    };
    bool lPassedFullRing = false;
    bool rPassedFullRing = false;
    TrianglesSortRes res = TrianglesSortRes::Undetermined;
    for ( ; tryNext || tryPrev; )
    {
        if ( tryNext )
            res = checkOther( true );
        if ( res != TrianglesSortRes::Undetermined )
            return res;
        if ( tryPrev )
            res = checkOther( false );
        if ( res != TrianglesSortRes::Undetermined )
            return res;

        if ( !lPassedFullRing && ( lNext == il.intersectionId || lPrev == il.intersectionId ) )
            lPassedFullRing = true;
        if ( !rPassedFullRing && ( rNext == ir.intersectionId || rPrev == ir.intersectionId ) )
            rPassedFullRing = true;

        if ( lPassedFullRing && rPassedFullRing )
            return TrianglesSortRes::Undetermined; // both contours passed a round, so break infinite loop
    }

    return res;
}

// baseEdge - cutting edge representation with orientation of first intersection
std::function<bool( const EdgeIntersectionData&, const EdgeIntersectionData& )> getLessFunc(
    const MeshTopology& tp,
    const std::vector<double>& dots, EdgeId baseEdge, const SortIntersectionsData* sortData )
{
    if ( !sortData )
    {
        return [&]( const EdgeIntersectionData& l, const EdgeIntersectionData& r ) -> bool
        {
            return dots[l.beforeSortIndex] < dots[r.beforeSortIndex];
        };
    }
    // sym baseEdge if other is not A:
    // if other is A intersection edge is going inside - out
    // otherwise it is going outside - in
    return[&tp, &dots, sortData, baseEdgeOr = sortData->isOtherA ? baseEdge : baseEdge.sym()]
    ( const EdgeIntersectionData& l, const EdgeIntersectionData& r ) -> bool
    {
        const auto & il = l.interOnEdge;
        const auto & ir = r.interOnEdge;

        FaceId fl = sortData->contours[il.contourId][il.intersectionId].tri();
        FaceId fr = sortData->contours[ir.contourId][ir.intersectionId].tri();
        EdgeId el = sortData->contours[il.contourId][il.intersectionId].edge;
        EdgeId er = sortData->contours[ir.contourId][ir.intersectionId].edge;
        assert( el.undirected() == baseEdgeOr.undirected() );
        assert( er.undirected() == baseEdgeOr.undirected() );

        // try sort by faces (topology)
        TrianglesSortRes res = sortTrianglesSymmetrical( *sortData, el, er, fl, fr, baseEdgeOr, EdgeSortState::Straight );
        if ( res != TrianglesSortRes::Undetermined )
            return res == TrianglesSortRes::Left;

        // try sort by next/prev intersections (topology)
        res = sortPropagateContour( tp, *sortData, il, ir, baseEdgeOr );
        if ( res != TrianglesSortRes::Undetermined )
            return res == TrianglesSortRes::Left;

        // try sort by geometry
        return dots[l.beforeSortIndex] < dots[r.beforeSortIndex];
    };
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
    PreCutResult res;
    res.paths.resize( contours.size() );
    res.oldEdgesInfo.resize( contours.size() );
    res.removedFaces.resize( contours.size() );
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

void sortEdgeInfo( const Mesh& mesh, const OneMeshContours& contours, EdgeData& edgeData,
    const SortIntersectionsData* sortData ) // it will probably be useful for precise sorting
{
    assert( !edgeData.empty() );
    const auto& intInfo = edgeData.front().interOnEdge;
    EdgeId baseEdge = std::get<EdgeId>( contours[intInfo.contourId].intersections[intInfo.intersectionId].primitiveId );

    std::vector<double> dotProds( edgeData.size() );
    Vector3d orgPoint{ mesh.orgPnt( baseEdge ) };
    auto abVec = Vector3d{ mesh.destPnt( baseEdge ) } - orgPoint;
    for ( int i = 0; i < edgeData.size(); ++i )
        dotProds[i] = dot( Vector3d{ mesh.points[edgeData[i].newVert] } - orgPoint, abVec );

    std::sort( edgeData.begin(), edgeData.end(), getLessFunc( mesh.topology, dotProds, baseEdge, sortData ) );

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

Expected<OneMeshContours> convertMeshTriPointsSurfaceOffsetToMeshContours( const Mesh& mesh, const std::vector<MeshTriPoint>& meshTriPoints, float isoValue, SearchPathSettings searchSettings )
{
    return convertMeshTriPointsSurfaceOffsetToMeshContours( mesh, meshTriPoints,
        [isoValue] ( int )
    {
        return isoValue;
    }, searchSettings );
}

Expected<OneMeshContours> convertMeshTriPointsSurfaceOffsetToMeshContours( const Mesh& mesh, const std::vector<MeshTriPoint>& meshTriPoints,
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
    auto initCutContourRes = convertMeshTriPointsToMeshContour( mesh, meshTriPoints, searchSettings, mmOffset.min == mmOffset.max ? nullptr : &pivotIndices );
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
    VertBitSet startVertices( meshAfterCut.topology.lastValidVert() + 1 );
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
    VertId currentId = mesh.topology.lastValidVert() + 1;
    for ( int i = 0; i < initCutContours.intersections.size(); ++i )
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
        resI.closed = true;
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
        resI.intersections.back() = resI.intersections.front();
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
    fixOrphans( mesh, preRes.paths, preRes.removedFaces, params.new2OldMap, params.new2oldEdgesMap );

    res.fbsWithContourIntersections = getBadFacesAfterCut( mesh.topology, preRes, preRes.removedFaces );
    if ( params.forceFillMode == CutMeshParameters::ForceFill::None && res.fbsWithContourIntersections.any() )
        return res;

    // find one edge for every hole to fill
    Timer t( "find edge per hole" );
    EdgeBitSet allHoleEdges( mesh.topology.edgeSize() );
    std::vector<EdgeId> holeRepresentativeEdges;
    std::vector<FaceId> oldFaces; // of corresponding holeRepresentativeEdges
    const bool needOldFaces = params.new2OldMap || params.new2oldEdgesMap;
    auto addHoleDesc = [&]( EdgeId e, FaceId oldf )
    {
        if ( allHoleEdges.test( e ) )
            return;
        holeRepresentativeEdges.push_back( e );
        if ( needOldFaces )
            oldFaces.push_back( oldf );
        for ( auto ei : leftRing( mesh.topology, e ) )
        {
            [[maybe_unused]] auto v = allHoleEdges.test_set( ei );
            assert( !v );
        }
    };
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
            if ( oldEdgesInfo[edgeId].hasLeft && !mesh.topology.left( path[edgeId] ).valid() )
                addHoleDesc( path[edgeId], oldf );
            if ( oldEdgesInfo[edgeId].hasRight && !mesh.topology.right( path[edgeId] ).valid() )
                addHoleDesc( path[edgeId].sym(), oldf );
        }
    }
    t.finish();

    // prepare in parallel the plan to fill every contour
    auto fillPlans = getPlanarHoleFillPlans( mesh, holeRepresentativeEdges );

    // fill contours
    t.restart( "run TriangulateContourPlans" );
    int numTris = 0;
    for ( const auto & plan : fillPlans )
        numTris += plan.numTris;
    const auto expectedTotalTris = mesh.topology.faceSize() + numTris;

    mesh.topology.faceReserve( expectedTotalTris );
    if ( params.new2OldMap )
        params.new2OldMap->reserve( expectedTotalTris );

    for ( size_t i = 0; i < holeRepresentativeEdges.size(); ++i )
        executeTriangulateContourPlan( mesh, holeRepresentativeEdges[i], fillPlans[i], 
            needOldFaces ? oldFaces[i] : FaceId{}, params.new2OldMap, params.new2oldEdgesMap );

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
        maybeOneMeshContours[ic] = convertMeshTriPointsToMeshContour( mesh, surfaceLine );
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
    auto sideFbv = fillContourLeft( mesh.topology, cutRes.resultCut );
    return sideFbv;
}

} //namespace MR
