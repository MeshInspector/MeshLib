#include "MRContoursCut.h"
#include "MRAffineXf3.h"
#include "MRMesh.h"
#include "MRTriangleIntersection.h"
#include "MRMeshTopology.h"
#include "MRMeshDelone.h"
#include "MRRingIterator.h"
#include "MRMeshFillHole.h"
#include "MRTimer.h"
#include "MRPrecisePredicates3.h"
#include "MRSurfacePath.h"
#include "MRMeshBuilder.h"
#include "MRBox.h"
#include "MRFillContour.h"
#include "MRGTest.h"
#include "MRMeshComponents.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRTBB.h"
#include <parallel_hashmap/phmap.h>
#include <numeric>

namespace MR
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

struct EdgeData
{
    std::vector<IntersectionData> intersections;
    std::vector<VertId> newVerts;
    std::vector<EdgeId> orgEdgeInLeftTri;
};

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
struct PreCutResult
{
    Vector<EdgeData, UndirectedEdgeId> edgeData;
    std::vector<EdgePath> paths;
    FullRemovedFacesInfo removedFaces;
    std::vector<std::vector<PathsEdgeIndex>> oldEdgesInfo;
};

bool isClosed( const ContinuousContour& contour )
{
    return contour.size() > 1 &&
        contour.front().isEdgeATriB == contour.back().isEdgeATriB &&
        contour.front().edge.undirected() == contour.back().edge.undirected() &&
        contour.front().tri == contour.back().tri;
}

enum class TrianglesSortRes
{
    Undetermined, // triangles positions cannot be determined
    Left,         // second triangle is form left side of oriented ABC
    Right         // second triangle is form right side of oriented ABC
};

void preparePreciseVerts( const SortIntersectionsData& sortData, VertId* verts, PreciseVertCoords* preciseVerts, int n )
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

// baseEdge - cutting edge representation with orientation of first intersection
std::function<bool( int, int )> getLessFunc( const EdgeData& edgeData, const std::vector<double>& dots, EdgeId baseEdge, const SortIntersectionsData* sortData )
{
    if ( !sortData )
    {
        return [&]( int l, int r ) -> bool
        {
            return dots[l] < dots[r];
        };
    }
    // sym baseEdge if other is not A:
    // if other is A intersection edge is going inside - out
    // otherwise it is going outside - in
    return[&edgeData, &dots, sortData, baseEdgeOr = sortData->isOtherA ? baseEdge : baseEdge.sym()]( int l, int r ) -> bool
    {
        const auto & il = edgeData.intersections[l];
        const auto & ir = edgeData.intersections[r];

        FaceId fl = sortData->contours[il.contourId][il.intersectionId].tri;
        FaceId fr = sortData->contours[ir.contourId][ir.intersectionId].tri;
        EdgeId el = sortData->contours[il.contourId][il.intersectionId].edge;
        EdgeId er = sortData->contours[ir.contourId][ir.intersectionId].edge;
        assert( el.undirected() == baseEdgeOr.undirected() );
        assert( er.undirected() == baseEdgeOr.undirected() );
        TrianglesSortRes res = sortTriangles( *sortData, fl, fr );
        if ( res != TrianglesSortRes::Undetermined )
            return ( el == baseEdgeOr ) == ( res == TrianglesSortRes::Left );

        res = sortTriangles( *sortData, fr, fl );
        if ( res != TrianglesSortRes::Undetermined )
            return ( er == baseEdgeOr ) == ( res == TrianglesSortRes::Right );

        return dots[l] < dots[r];
    };
}

void subdivideLoneContours( Mesh& mesh, const OneMeshContours& contours, FaceMap* new2oldMap /*= nullptr */ )
{
    MR_TIMER;
    MR_MESH_WRITER( mesh );
    phmap::flat_hash_map<int, std::vector<int>> face2contoursMap;
    for ( int i = 0; i < contours.size(); ++i )
    {
        FaceId f = std::get<FaceId>( contours[i].intersections.front().primitiveId );
        face2contoursMap[f].push_back( i );
    }
    for ( auto& [faceId, conts] : face2contoursMap )
    {
        Vector3f massCenter;
        int counter = 0;
        for ( const auto& cont : conts )
        {
            for ( const auto& p : contours[cont].intersections )
            {
                ++counter;
                massCenter += p.coordinate;
            }
        }
        massCenter /= float( counter );

        EdgeId e0, e1, e2;
        FaceId f = FaceId( faceId );
        e0 = mesh.topology.edgePerFace()[f];
        e1 = mesh.topology.prev( e0.sym() );
        e2 = mesh.topology.prev( e1.sym() );
        mesh.topology.setLeft( e0, {} );
        VertId newV = mesh.addPoint( massCenter );
        EdgeId en0 = mesh.topology.makeEdge();
        EdgeId en1 = mesh.topology.makeEdge();
        EdgeId en2 = mesh.topology.makeEdge();
        mesh.topology.setOrg( en0, newV );
        mesh.topology.splice( en0, en1 );
        mesh.topology.splice( en1, en2 );
        mesh.topology.splice( e0, en0.sym() );
        mesh.topology.splice( e1, en1.sym() );
        mesh.topology.splice( e2, en2.sym() );
        FaceId nf0 = mesh.topology.addFaceId();
        FaceId nf1 = mesh.topology.addFaceId();
        FaceId nf2 = mesh.topology.addFaceId();
        mesh.topology.setLeft( en0, nf0 );
        mesh.topology.setLeft( en1, nf1 );
        mesh.topology.setLeft( en2, nf2 );
        if ( new2oldMap )
        {
            new2oldMap->autoResizeAt( nf2 ) = f;
            ( *new2oldMap )[nf1] = ( *new2oldMap )[nf0] = f;
        }
    }
}

OneMeshContours getOneMeshIntersectionContours( const Mesh& meshA, const Mesh& meshB, const ContinuousContours& contours, bool getMeshAIntersections,
    const CoordinateConverters& converters, const AffineXf3f* rigidB2A /*= nullptr */ )
{
    MR_TIMER;
    OneMeshContours res;

    std::function<Vector3f( const Vector3f& coord, bool meshA )> getCoord;

    if ( !rigidB2A )
    {
        getCoord = []( const Vector3f& coord, bool )
        {
            return coord;
        };
    }
    else
    {
        getCoord = [xf = *rigidB2A]( const Vector3f& coord, bool meshA )
        {
            return meshA ? coord : xf( coord );
        };
    }
    AffineXf3f inverseXf;
    if ( rigidB2A )
        inverseXf = rigidB2A->inverse();
    const auto& mainMesh = getMeshAIntersections ? meshA : meshB;
    const auto& otherMesh = getMeshAIntersections ? meshB : meshA;
    res.resize( contours.size() );
    for ( int j = 0; j < contours.size(); ++j )
    {
        auto& curOutContour = res[j].intersections;
        const auto& curInContour = contours[j];
        res[j].closed = isClosed( curInContour );
        curOutContour.resize( curInContour.size() );

        tbb::parallel_for( tbb::blocked_range<size_t>( 0, curInContour.size() ),
            [&]( const tbb::blocked_range<size_t>& range )
        {
            Vector3f a, b, c, d, e;
            for ( size_t i = range.begin(); i < range.end(); ++i )
            {
                const auto& inIntersection = curInContour[i];
                auto& outIntersection = curOutContour[i];

                bool edgeMain = getMeshAIntersections == inIntersection.isEdgeATriB;
                if ( edgeMain )
                {
                    outIntersection.primitiveId = inIntersection.edge;
                    otherMesh.getTriPoints( inIntersection.tri, a, b, c );
                    d = mainMesh.orgPnt( inIntersection.edge );
                    e = mainMesh.destPnt( inIntersection.edge );
                }
                else
                {
                    outIntersection.primitiveId = inIntersection.tri;
                    mainMesh.getTriPoints( inIntersection.tri, a, b, c );
                    d = otherMesh.orgPnt( inIntersection.edge );
                    e = otherMesh.destPnt( inIntersection.edge );
                }
                // always calculate in mesh A space
                outIntersection.coordinate = findTriangleSegmentIntersectionPrecise( 
                    getCoord( a, !inIntersection.isEdgeATriB ),
                    getCoord( b, !inIntersection.isEdgeATriB ),
                    getCoord( c, !inIntersection.isEdgeATriB ),
                    getCoord( d, inIntersection.isEdgeATriB ),
                    getCoord( e, inIntersection.isEdgeATriB ), converters );

                if ( !getMeshAIntersections && rigidB2A )
                    outIntersection.coordinate = inverseXf( outIntersection.coordinate );
            }
        } );
    }
    return res;
}

// finds FaceId if face, shared among vid eid and mtp, if there is no one returns left(mtp.e)
FaceId findSharedFace( const MeshTopology& topology, VertId vid, EdgeId eid, const MeshTriPoint& mtp )
{
    auto mtpEdgeRep = mtp.onEdge( topology );
    if ( !mtpEdgeRep )
        return topology.left( mtp.e );

    if ( topology.dest( eid ) == vid )
        eid = eid.sym();

    auto mtpVertRep = mtpEdgeRep->inVertex( topology );
    if ( mtpVertRep.valid() )
    {
        if ( topology.dest( topology.next( eid ) ) == mtpVertRep )
            return topology.left( eid );
        else if ( topology.dest( topology.prev( eid ) ) == mtpVertRep )
            return topology.right( eid );
    }
    else
    {
        auto mtpEUndir = mtpEdgeRep->e.undirected();
        if ( topology.next( eid ).undirected() == mtpEUndir )
            return topology.left( eid );
        else if ( topology.prev( eid ).undirected() == mtpEUndir )
            return topology.right( eid );

        eid = eid.sym();
        if ( topology.next( eid ).undirected() == mtpEUndir )
            return topology.left( eid );
        else if ( topology.prev( eid ).undirected() == mtpEUndir )
            return topology.right( eid );
    }
    return topology.left( mtp.e );
}

enum class CenterInterType
{
    Common, // result of centerInter should be inserted as is
    VertsAreSame, // three points (prev center next) are the same vertex (only one of them should be inserted)
    SameEdgesClosePos // three points have close positions so only one (inside face intersection) is inserted, not to break float based sorting
};

// `centralIntersection` function extension for cases when one of (or both) `prev` `next` is (are) in face
// `centralIntersection` - function to get information about mesh tri point intersection between two surfaces paths and process it correctly
std::optional<OneMeshIntersection> centralIntersectionForFaces( const Mesh& mesh, const OneMeshIntersection& prev, const MeshTriPoint& curr, const OneMeshIntersection& next )
{
    auto optMeshEdgePoint = curr.onEdge( mesh.topology );
    if ( !optMeshEdgePoint ) // curr on face
    {
#ifndef NDEBUG
        if ( prev.primitiveId.index() == OneMeshIntersection::Face )
        {
            assert( std::get<FaceId>( prev.primitiveId ) == mesh.topology.left( curr.e ) );
        }
        if ( next.primitiveId.index() == OneMeshIntersection::Face )
        {
            assert( std::get<FaceId>( next.primitiveId ) == mesh.topology.left( curr.e ) );
        }
#endif
        return OneMeshIntersection{ mesh.topology.left( curr.e ),mesh.triPoint( curr ) };
    }
    else
    {
        auto v = optMeshEdgePoint->inVertex( mesh.topology );
        if ( v ) // curr in vertex
        {
            if ( prev.primitiveId.index() == OneMeshIntersection::Vertex )
            {
                auto pV = std::get<VertId>( prev.primitiveId );
                if ( pV == v )
                    return {};
            }
            if ( next.primitiveId.index() == OneMeshIntersection::Vertex )
            {
                auto nV = std::get<VertId>( next.primitiveId );
                if ( nV == v )
                    return {};
            }
            return OneMeshIntersection{ v,mesh.points[v] };
        }

        auto e = optMeshEdgePoint->e;

        // only need that prev and next are on different sides of curr
        if ( prev.primitiveId.index() == OneMeshIntersection::Face )
        {
            auto pFId = std::get<FaceId>( prev.primitiveId );
            if ( mesh.topology.right( e ) != pFId )
            {
                e = e.sym(); // orient correct to be consistent with path
                assert( mesh.topology.right( e ) == pFId );
            }

            if ( next.primitiveId.index() == OneMeshIntersection::Vertex )
            {
                auto nVId = std::get<VertId>( next.primitiveId );
                if ( mesh.topology.dest( mesh.topology.next( e ) ) != nVId )
                {
#ifndef NDEBUG
                    VertId pV[3];
                    mesh.topology.getTriVerts( pFId, pV );
                    assert( nVId == pV[0] || nVId == pV[1] || nVId == pV[2] );
#endif
                    return {};
                }
                else
                {
                    return OneMeshIntersection{ e,mesh.edgePoint( *optMeshEdgePoint ) };
                }
            }
            else if ( next.primitiveId.index() == OneMeshIntersection::Edge )
            {
                auto nEUId = std::get<EdgeId>( next.primitiveId ).undirected();
                if ( mesh.topology.next( e ).undirected() != nEUId &&
                     mesh.topology.prev( e.sym() ).undirected() != nEUId )
                {
                    assert( nEUId == e.undirected() || nEUId == mesh.topology.prev( e ).undirected() || nEUId == mesh.topology.next( e.sym() ).undirected() );
                    return {};
                }
                else
                {
                    return OneMeshIntersection{ e,mesh.edgePoint( *optMeshEdgePoint ) };
                }
            }
            else // if face
            {
                auto nFId = std::get<FaceId>( next.primitiveId );
                if ( pFId == nFId )
                {
                    return {};
                }
                else
                {
                    assert( nFId == mesh.topology.left( e ) );
                    return OneMeshIntersection{ e,mesh.edgePoint( *optMeshEdgePoint ) };
                }
            }
        }
        else // if next == face and prev is not face, symetric
        {
            auto nFId = std::get<FaceId>( next.primitiveId );
            if ( mesh.topology.left( e ) != nFId )
            {
                e = e.sym();
                assert( mesh.topology.left( e ) == nFId );
            }

            if ( prev.primitiveId.index() == OneMeshIntersection::Vertex )
            {
                auto pVId = std::get<VertId>( prev.primitiveId );
                if ( mesh.topology.dest( mesh.topology.prev( e ) ) != pVId )
                {
#ifndef NDEBUG
                    VertId nV[3];
                    mesh.topology.getTriVerts( nFId, nV );
                    assert( pVId == nV[0] || pVId == nV[1] || pVId == nV[2] );
#endif
                    return {};
                }
                else
                {
                    return OneMeshIntersection{ e,mesh.edgePoint( *optMeshEdgePoint ) };
                }

            }
            else // if edge
            {
                auto pEUId = std::get<EdgeId>( prev.primitiveId ).undirected();
                if ( mesh.topology.prev( e ).undirected() != pEUId &&
                     mesh.topology.next(e.sym() ).undirected() != pEUId )
                {
                    assert( pEUId == e.undirected() || pEUId == mesh.topology.next( e ).undirected() || pEUId == mesh.topology.prev( e.sym() ).undirected() );
                    return {};
                }
                else
                {
                    return OneMeshIntersection{ e,mesh.edgePoint( *optMeshEdgePoint ) };
                }
            }
        }
    }
}

// function to get information about mesh tri point intersection between two surfaces paths
// and process it correctly
std::optional<OneMeshIntersection> centralIntersection( const Mesh& mesh, const OneMeshIntersection& prev, const MeshTriPoint& curr, const OneMeshIntersection& next,
                                                float closeEdgeEps,
                                                CenterInterType& type )
{
    MR_TIMER;
    type = CenterInterType::Common;
    if ( prev.primitiveId.index() == OneMeshIntersection::Face || next.primitiveId.index() == OneMeshIntersection::Face )
        return centralIntersectionForFaces( mesh, prev, curr, next );

    const auto& topology = mesh.topology;
    if ( prev.primitiveId.index() == OneMeshIntersection::Vertex )
    {
        auto pVId = std::get<VertId>( prev.primitiveId );
        if ( next.primitiveId.index() == OneMeshIntersection::Vertex )
        {
            auto nVId = std::get<VertId>( next.primitiveId );
            if ( nVId == pVId )
            {
                type = CenterInterType::VertsAreSame;
                return {};
            }
            for ( auto e : orgRing( topology, pVId ) )
            {
                if ( topology.dest( e ) == nVId )
                    return {};
            }
        }
        else if ( next.primitiveId.index() == OneMeshIntersection::Edge )
        {
            auto nEId = std::get<EdgeId>( next.primitiveId );
            if ( topology.dest( topology.prev( nEId ) ) == pVId ) // correct orientation
                return {};
            if ( topology.dest( topology.next( nEId ) ) == pVId ) // incorrect orientation
                return {};
            if ( topology.dest( nEId ) == pVId || topology.org( nEId ) == pVId )
                return OneMeshIntersection{findSharedFace(topology,pVId,nEId,curr),mesh.triPoint( curr )};
        }
        else
        {
            assert( false );
        }

        auto edgeOp = curr.onEdge( topology );
        assert( edgeOp );
        auto vid = edgeOp->inVertex( topology );
        if ( vid.valid() )
            return OneMeshIntersection{vid,mesh.points[vid]};
        if ( topology.dest( topology.prev( edgeOp->e ) ) == pVId )
            return OneMeshIntersection{edgeOp->e,mesh.edgePoint( *edgeOp )};
        else
            return OneMeshIntersection{edgeOp->e.sym(),mesh.edgePoint( *edgeOp )};
    }
    else if ( prev.primitiveId.index() == OneMeshIntersection::Edge )
    {
        auto pEId = std::get<EdgeId>( prev.primitiveId );
        if ( next.primitiveId.index() == OneMeshIntersection::Vertex )
        {
            auto nVId = std::get<VertId>( next.primitiveId );
            if ( topology.dest( topology.next( pEId ) ) == nVId )
                return {};
            if ( topology.dest( pEId ) == nVId || topology.org( pEId ) == nVId )
                return OneMeshIntersection{findSharedFace( topology,nVId,pEId,curr ),mesh.triPoint( curr )};
        }
        else if ( next.primitiveId.index() == OneMeshIntersection::Edge )
        {
            auto nEId = std::get<EdgeId>( next.primitiveId );
            if ( nEId.undirected() == pEId.undirected() )
            {
                FaceId currF = findSharedFace( mesh.topology, mesh.topology.dest( nEId ), nEId, curr );
                assert( currF );
                auto coordDif = ( next.coordinate - prev.coordinate ).length();
                if ( coordDif < closeEdgeEps )
                {
                    type = CenterInterType::SameEdgesClosePos;
                    if ( currF == topology.left( nEId ) )
                        currF = topology.right( nEId );
                    else if ( currF == topology.right( nEId ) )
                        currF = topology.left( nEId );
                }
                return OneMeshIntersection{currF,mesh.triPoint( curr )};
            }

            // consistent orientation:
            // topology.next( pEId ) == nEId || topology.prev( pEId.sym() ) == nEId.sym()
            // 
            // not consistent orientation:
            // it means that one of `prev` or `next` are form original MeshTriPoints vector - orientation should be fixed when this MeshTriPoint will be `curr` in this function
            // all mesh tri points will be curr in this function once so, output orientation should be correct
            if ( topology.next( pEId ).undirected() == nEId.undirected() || 
                 topology.prev( pEId.sym() ).undirected() == nEId.sym().undirected() )
            {
                // orientation is consistent or `next` orientation can be wrong
                auto edgeOp = curr.onEdge( topology );
                if ( edgeOp )
                    return {};
                return OneMeshIntersection{topology.left( curr.e ),mesh.triPoint( curr )};
            }
            else if ( topology.prev( pEId ).undirected() == nEId.undirected() ||
                      topology.next( pEId.sym() ).undirected() == nEId.sym().undirected() )
            {
                // `prev` orientation is wrong (rare case, only seen with first intersection only)
                auto edgeOp = curr.onEdge( topology );
                if ( edgeOp )
                    return {};
                return OneMeshIntersection{ topology.left( curr.e ),mesh.triPoint( curr ) };
            }
            // else statement means that `prev` and `next` do not share face 
            // only correct `curr` position is on the edge that share one face with `prev` and other face with `next`
        }
        else
        {
            assert( false );
        }

        auto edgeOp = curr.onEdge( topology );
        assert( edgeOp );
        auto vid = edgeOp->inVertex( topology );
        if ( vid.valid() )
            return OneMeshIntersection{vid,mesh.points[vid]};
        if ( topology.prev( edgeOp->e ) == pEId || topology.next( edgeOp->e.sym() ) == pEId.sym() )
            return OneMeshIntersection{edgeOp->e,mesh.edgePoint( *edgeOp )};
        else
            return OneMeshIntersection{edgeOp->e.sym(),mesh.edgePoint( *edgeOp )};
    }
    else
    {
        assert( false );
    }
    return {};
}

OneMeshIntersection intersectionFromMeshTriPoint( const Mesh& mesh, const MeshTriPoint& mtp )
{
    OneMeshIntersection res;
    res.coordinate = mesh.triPoint( mtp );
    auto e = mtp.onEdge( mesh.topology );
    if ( e )
    {
        auto v = e->inVertex( mesh.topology );
        if ( v )
            res.primitiveId = v;
        else
            res.primitiveId = e->e;
    }
    else
        res.primitiveId = mesh.topology.left( mtp.e );
    return res;
}

OneMeshContour convertMeshTriPointsToClosedContour( const Mesh& mesh, const std::vector<MeshTriPoint>& meshTriPointsOrg )
{
    MR_TIMER;
    // clear duplicates
    auto meshTriPoints = meshTriPointsOrg;
    std::vector<int> sameEdgeMTPs;
    Box3f box;
    for ( int i = 0; i < meshTriPoints.size(); ++i )
    {
        box.include( mesh.triPoint( meshTriPoints[i] ) );
        auto e1 = meshTriPoints[i].onEdge( mesh.topology );
        auto e2 = meshTriPoints[( i + 1 ) % meshTriPoints.size()].onEdge( mesh.topology );
        if ( !e1 || !e2 )
            continue;
        auto v1 = e1->inVertex( mesh.topology );
        auto v2 = e2->inVertex( mesh.topology );
        if ( v1.valid() && v2.valid() )
        {
            if ( v1 == v2 )
                sameEdgeMTPs.push_back( i );
        }
        else
        {
            if ( e1->e.undirected() == e2->e.undirected() )
                sameEdgeMTPs.push_back( i );
        }
    }
    for ( int i = int( sameEdgeMTPs.size() ) - 1; i >= 0; --i )
        meshTriPoints.erase( meshTriPoints.begin() + sameEdgeMTPs[i] );

    // build paths
    if ( meshTriPoints.size() < 2 )
        return {};
    OneMeshContour res;
    std::vector<OneMeshContour> surfacePaths( meshTriPoints.size() );
    for ( int i = 0; i < meshTriPoints.size(); ++i )
    {        
        auto sp = computeSurfacePath( mesh, meshTriPoints[i], meshTriPoints[( i + 1 ) % meshTriPoints.size()] );
        if ( !sp.has_value() )
            continue;
        auto partContours = convertSurfacePathsToMeshContours( mesh, { std::move( sp.value() ) } );
        assert( partContours.size() == 1 );
        surfacePaths[i] = partContours[0];
        if ( surfacePaths[i].intersections.size() == 1 )
        {
            // if lone make sure that prev MeshTriPoint is on the right (if on the left - sym)
            if ( surfacePaths[i].intersections[0].primitiveId.index() == OneMeshIntersection::Edge )
            {
                auto& edge = std::get<EdgeId>( surfacePaths[i].intersections[0].primitiveId );
                auto onEdge = meshTriPoints[i].onEdge( mesh.topology );
                if ( !onEdge )
                {
                    // if mtp is on face - make sure that it is right of edge
                    if ( mesh.topology.left( edge ) == mesh.topology.left( meshTriPoints[i].e ) )
                        edge = edge.sym();
                }
                else
                {
                    auto inVert = onEdge->inVertex( mesh.topology );
                    if ( !inVert )
                    {
                        // if mtp is on edge - make sure it is prev(e) or next(e.sym)
                        if ( mesh.topology.next( edge ).undirected() == onEdge->e.undirected() ||
                            mesh.topology.prev( edge.sym() ).undirected() == onEdge->e.undirected() )
                            edge = edge.sym();
                    }
                    else
                    {
                        // if mtp is in vert - make sure it is dest(prev(e))
                        if ( inVert == mesh.topology.dest( mesh.topology.next( edge ) ) )
                            edge = edge.sym();
                    }
                }
            }
        }
    }

    const float closeEdgeEps = std::numeric_limits<float>::epsilon() * box.diagonal();
    // add interjacent
    for ( int i = 0; i < meshTriPoints.size(); ++i )
    {
        auto& prevInter = surfacePaths[( i + int( meshTriPoints.size() ) - 1 ) % meshTriPoints.size()].intersections;
        auto& nextInter = surfacePaths[i].intersections;
        OneMeshIntersection lastPrev;
        OneMeshIntersection firstNext;
        if ( prevInter.empty() )
        {
            if ( !res.intersections.empty() )
                lastPrev = res.intersections.back();
            else
                lastPrev = intersectionFromMeshTriPoint( mesh, meshTriPoints[( i + int( meshTriPoints.size() ) - 1 ) % meshTriPoints.size()] );
        }
        else
            lastPrev = prevInter.back();
        if ( nextInter.empty() )
        {
            firstNext = intersectionFromMeshTriPoint( mesh, meshTriPoints[( i + 1 ) % meshTriPoints.size()] );
        }
        else
            firstNext = nextInter.front();
        CenterInterType type;
        auto centerInterOp = centralIntersection( mesh, lastPrev, meshTriPoints[i], firstNext, closeEdgeEps, type );
        if ( centerInterOp )
        {
            if ( type != CenterInterType::SameEdgesClosePos )
                res.intersections.push_back( *centerInterOp );
            else
            {
                if ( res.intersections.empty() )
                    prevInter.back() = *centerInterOp;
                else
                    res.intersections.back() = *centerInterOp;

            }
        }
        if ( !surfacePaths[i].intersections.empty() )
        {
            if ( type == CenterInterType::Common )
                res.intersections.insert( res.intersections.end(), surfacePaths[i].intersections.begin(), surfacePaths[i].intersections.end() );
            else
                res.intersections.insert( res.intersections.end(), surfacePaths[i].intersections.begin() + 1, surfacePaths[i].intersections.end() );
        }
    }
    res.intersections.push_back( res.intersections.front() );
    res.closed = true;
    return res;
}

OneMeshContour convertSurfacePathWithEndsToMeshContour( const MR::Mesh& mesh, const MeshTriPoint& start, const MR::SurfacePath& surfacePath, const MeshTriPoint& end )
{
    if ( surfacePath.empty() )
    {
        spdlog::warn( "Surface path is empty" );
        return {};
    }
#ifndef NDEBUG
    auto mtpCpy = start;
    auto pathMtpCpy = MeshTriPoint( surfacePath.front() );
    if ( !fromSameTriangle( mesh.topology, mtpCpy, pathMtpCpy ) )
    {
        spdlog::error( "Start point and first path point are not from the same face" );
        assert( false );
        return {};
    }
    mtpCpy = end;
    pathMtpCpy = MeshTriPoint( surfacePath.back() );
    if ( !fromSameTriangle( mesh.topology, mtpCpy, pathMtpCpy ) )
    {
        spdlog::error( "End point and last path point are not from the same face" );
        assert( false );
        return {};
    }
#endif
    auto startMep = start.onEdge( mesh.topology );
    auto endMep = end.onEdge( mesh.topology );
    OneMeshContour res;
    if ( startMep || endMep )
    {
        int shift = startMep ? 1 : 0;
        SurfacePath updatedPath( surfacePath.size() + shift + ( endMep ? 1 : 0 ) );
        if ( startMep )
            updatedPath.front() = *startMep;
        for ( int i = 0; i < surfacePath.size(); ++i )
            updatedPath[i + shift] = surfacePath[i];
        if ( endMep )
            updatedPath.back() = *endMep;
        res = convertSurfacePathsToMeshContours( mesh, { updatedPath } ).front();
    }
    else
    {
        res = convertSurfacePathsToMeshContours( mesh, { surfacePath } ).front();
    }
    if ( !startMep )
        res.intersections.insert( res.intersections.begin(), intersectionFromMeshTriPoint( mesh, start ) );
    if ( !endMep )
        res.intersections.push_back( intersectionFromMeshTriPoint( mesh, end ) );

    if ( res.intersections.front().primitiveId == res.intersections.back().primitiveId &&
         res.intersections.front().coordinate == res.intersections.back().coordinate )
        res.closed = true;

    return res;
}

// this function orient intersected edges to start from left part of path and end on right, also remove duplicates
SurfacePath formatSurfacePath( const MeshTopology& topology, const  SurfacePath& path )
{
    MR_TIMER;
    SurfacePath res;
    res.reserve( path.size() );
    VertId prevVId, vId, nextVId;
    UndirectedEdgeId prevUEId,uEId;
    // this cycle leaves only unique intersections removing duplicates
    for ( int i = 0; i < path.size(); ++i )
    {
        vId = path[i].inVertex( topology );
        if ( vId.valid() && vId == prevVId )
            continue;

        if ( !vId.valid() && i != 0 && i != int( path.size() ) - 1 && topology.isBdEdge(path[i].e) )
            continue;

        uEId = path[i].e.undirected();
        if ( !vId.valid() && prevUEId == uEId )
            continue;

        if ( vId.valid() )
        {
            while ( !res.empty() && !res.back().inVertex() )
            {
                EdgeId dirE = res.back().e;
                if ( topology.org( dirE ) == vId || topology.dest( dirE ) == vId )
                    res.pop_back();
                else
                    break;
            }
        }
        else if ( !vId.valid() && prevVId.valid() )
        {
            EdgeId dirE( uEId );
            if ( topology.org( dirE ) == prevVId || topology.dest( dirE ) == prevVId )
            {
                continue;
            }
        }

        res.push_back( path[i] );
        prevVId = vId;
        prevUEId = vId.valid() ? UndirectedEdgeId{} : uEId;
    }

    if ( res.size() < 2 )
        return res;

    int i = 0;
    for ( i = 0; i < res.size(); ++i )
        if ( !res[i].inVertex() )
            break;

    if ( i == 0 )
    {
        auto& firstInter = res[i];
        const auto& nextInter = res[i + 1];
        nextVId = nextInter.inVertex( topology );
        if ( !nextVId.valid() )
        {
            auto nextUndirected = nextInter.e.undirected();
            if ( nextUndirected != topology.next( firstInter.e ).undirected() &&
                 nextUndirected != topology.prev( firstInter.e.sym() ).undirected() )
                firstInter = firstInter.sym();
        }
        else
        {
            if ( topology.dest( topology.next( firstInter.e ) ) != nextVId &&
                 topology.dest( topology.prev( firstInter.e.sym() ) ) != nextVId )
                firstInter = firstInter.sym();
        }
        ++i;
    }
    for ( ; i < res.size(); ++i )
    {
        const auto& prevInter = res[i - 1];
        auto& inter = res[i];
        if ( inter.inVertex() )
            continue;

        prevVId = prevInter.inVertex( topology );
        if ( prevVId.valid() )
        {
            if ( topology.dest( topology.prev( inter.e ) ) != prevVId &&
                 topology.dest( topology.next( inter.e.sym() ) ) != prevVId )
                inter = inter.sym();
        }
        else
        {
            auto prevUndirected = prevInter.e.undirected();
            if ( prevUndirected != topology.prev( inter.e ).undirected() &&
                 prevUndirected != topology.next( inter.e.sym() ).undirected() )
                inter = inter.sym();
        }
    }

    return res;
}

OneMeshContours convertSurfacePathsToMeshContours( const Mesh& mesh, const std::vector<SurfacePath>& surfacePaths )
{
    MR_TIMER;
    OneMeshContours res;

    res.resize( surfacePaths.size() );

    for ( int j = 0; j < surfacePaths.size(); ++j )
    {
        auto& curOutContour = res[j].intersections;
        const auto curInContour = formatSurfacePath( mesh.topology, surfacePaths[j] );
        res[j].closed = false;
        if ( curInContour.size() > 1 )
        {
            const auto& front = curInContour.front();
            const auto& back = curInContour.back();
            auto vF = front.inVertex( mesh.topology );
            auto vB = back.inVertex( mesh.topology );
            if ( vF.valid() && vF == vB )
                res[j].closed = true;
            else if ( !vF && !vB && front.e == back.e && front.a == back.a )
                res[j].closed = true;
        }

        curOutContour.resize( curInContour.size() );
        tbb::parallel_for( tbb::blocked_range<size_t>( 0, curInContour.size() ),
            [&]( const tbb::blocked_range<size_t>& range )
        {
            VertId vid;
            for ( size_t i = range.begin(); i < range.end(); ++i )
            {
                const auto& inIntersection = curInContour[i];
                auto& outIntersection = curOutContour[i];
                vid = inIntersection.inVertex( mesh.topology );
                if ( vid.valid() )
                    outIntersection.primitiveId = vid;
                else
                    outIntersection.primitiveId = inIntersection.e;
                outIntersection.coordinate = mesh.edgePoint( inIntersection );
            }
        } );
    }
    return res;
}

// sets left face of given edge invalid and saves info about its old left ring
void invalidateFace( MeshTopology& topology, FullRemovedFacesInfo& removedFaceInfo, int contId, int interId, EdgeId leftEdge, size_t oldEdgesSize )
{
    auto thisFace = topology.left( leftEdge );
    if ( thisFace.valid() )
    {
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
}

// back-iterate removed face info to find correct splice edge for vert->face->vert like paths (second vert should find correct face edge for splice)
EdgeId iterateRemovedFacesInfoToFindLeftEdge( const MeshTopology& topology, const FullRemovedFacesInfo& removedFaces, int contId, int interId, FaceId f, VertId v )
{
    MR_TIMER
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
            bool isVert = inter.primitiveId.index() == OneMeshIntersection::Vertex;
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
                isNextVert = nextPrimId.index() == OneMeshIntersection::Vertex;

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
                        if ( nextPrimId.index() == OneMeshIntersection::Edge )
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
                    if ( prevInterPrimId.index() == OneMeshIntersection::Edge )
                    {
                        auto e = std::get<EdgeId>( prevInterPrimId );
                        invalidateFace( mesh.topology, res.removedFaces, contourId, intersectionId - 1, mesh.topology.next( e ).sym(), oldEdgesSize );
                        mesh.topology.splice( mesh.topology.next( e ).sym(), path[intersectionId - 1].sym() );
                    }
                    else if ( prevInterPrimId.index() == OneMeshIntersection::Face )
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
                            prevFaceIncidentLeft = iterateRemovedFacesInfoToFindLeftEdge( mesh.topology, res.removedFaces, contourId, intersectionId, f, newVertId );
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
            if ( newVertId.valid() && inter.primitiveId.index() == OneMeshIntersection::Edge )
            {
                EdgeId thisEdge = std::get<EdgeId>( inter.primitiveId );
                auto& edgeData = res.edgeData.autoResizeAt( thisEdge.undirected() );
                edgeData.orgEdgeInLeftTri.push_back( newEdgeId );
                edgeData.newVerts.push_back( newVertId );
                edgeData.intersections.push_back( {ContourId( contourId ),IntersectionId( intersectionId )} );
                // fill removed tris
                removedFacesInfo[intersectionId].f = mesh.topology.left( thisEdge );
            }
            // fill removed tris
            if ( inter.primitiveId.index() == OneMeshIntersection::Face )
                removedFacesInfo[intersectionId].f = std::get<FaceId>( inter.primitiveId );

        }
        if ( closed )
        {
            if ( inContour.back().primitiveId.index() != OneMeshIntersection::Vertex )
                mesh.topology.splice( path.back().sym(), path.front() );
        }
        else
        {
            if ( inContour.back().primitiveId.index() != OneMeshIntersection::Vertex )
            {
                assert( newVertId.valid() );
                mesh.topology.setOrg( path.back().sym(), newVertId );
            }
        }
    }
    return res;
}

void triangulateContour( Mesh& mesh, EdgeId e, FaceId oldFace, FaceMap* new2OldMap )
{
    assert( oldFace.valid() );
    if ( !new2OldMap )
        return fillHole( mesh, e, { std::make_unique<PlaneNormalizedFillMetric>( mesh,e ) } );

    FaceBitSet newFaces;
    fillHole( mesh, e, { std::make_unique<PlaneNormalizedFillMetric>( mesh,e ), &newFaces} );
    for ( auto f : newFaces )
        new2OldMap->autoResizeAt( f ) = oldFace;
}

/* this function triangulate holes where first and last edge are the same but sym
       / \
     /___  \
   /         \      hole starts with central edge going to triangle and finishes with it (shape can be any, not only triangle)
 /_____________\

edges should be already cut */
void fixOrphans( Mesh& mesh, const std::vector<EdgePath>& paths, const FullRemovedFacesInfo& removedFaces, FaceMap* new2OldMap )
{

    auto fixOrphan = [&]( EdgeId e, FaceId oldF )
    {
        if ( mesh.topology.left( e ).valid() ||
             mesh.topology.right( e ).valid() )
            return;

        auto next = mesh.topology.next( e.sym() );
        auto newEdge = mesh.topology.makeEdge();
        mesh.topology.splice( e, newEdge );
        mesh.topology.splice( next.sym(), newEdge.sym() );

        triangulateContour( mesh, e, oldF, new2OldMap );
        triangulateContour( mesh, e.sym(), oldF, new2OldMap );
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

void debugSortingInfo( EdgeId baseE,
                       const EdgeData& edgeData,
                       const std::vector<int>& res,
                       const std::vector<float>& dotProds,
                       const SortIntersectionsData* sortData )
{
    const auto& verts = edgeData.newVerts;
    if ( verts.size() > 1 )
    {
        bool edgeInfoPrinted{false};
        for ( int i = 0; i + 1 < res.size(); ++i )
            //if ( dotProds[res[i]] == dotProds[res[i + 1]] )
        {
            if ( !edgeInfoPrinted )
            {
                spdlog::info( "Edge {}", baseE );
                edgeInfoPrinted = true;
            }
            if ( sortData )
            {
                FaceId f1 = sortData->contours[edgeData.intersections[res[i]].contourId][edgeData.intersections[res[i]].intersectionId].tri;
                FaceId f2 = sortData->contours[edgeData.intersections[res[i + 1]].contourId][edgeData.intersections[res[i + 1]].intersectionId].tri;

                auto sharedEdge = sortData->otherMesh.topology.sharedEdge( f1, f2 );
                spdlog::info( "  {}", dotProds[res[i + 1]] - dotProds[res[i]] );
                spdlog::info( "   shared: ", sharedEdge );
            }
        }
    }
}

std::vector<int> sortIntersectionsAlongBaseEdge( const Mesh& mesh, EdgeId baseE,
                                                 const EdgeData& edgeData, 
                                                 const SortIntersectionsData* sortData ) // it will probably be useful for precise sorting
{
    const auto& verts = edgeData.newVerts;
    if ( verts.empty() )
        return {};
    std::vector<int> res( verts.size() );
    std::iota( res.begin(), res.end(), 0 );

    std::vector<double> dotProds( verts.size() );
    Vector3d orgPoint{ mesh.orgPnt( baseE ) };
    auto abVec = Vector3d{ mesh.destPnt( baseE ) } - orgPoint;
    for ( int i = 0; i < verts.size(); ++i )
        dotProds[i] = dot( Vector3d{ mesh.points[verts[i]] } - orgPoint, abVec );

    std::sort( res.begin(), res.end(), getLessFunc( edgeData, dotProds, baseE, sortData ) );

    // DEBUG Output
    //debugSortingInfo( baseE, edgeData, res, dotProds, sortData );

    return res;
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
                 const SortIntersectionsData* sortData,
                 FaceMap* new2OldMap )
{
    assert( !edgeData.intersections.empty() );

    const auto& intInfo = edgeData.intersections[0];
    EdgeId baseEdge = std::get<EdgeId>( contours[intInfo.contourId].intersections[intInfo.intersectionId].primitiveId );

    // will need this to restore lost face on first or last intersection (only for open contours)
    FaceId oldLeft = mesh.topology.left( baseEdge );
    FaceId oldRight = mesh.topology.left( baseEdge.sym() );

    // remove incident faces
    mesh.topology.setLeft( baseEdge, FaceId{} );
    mesh.topology.setLeft( baseEdge.sym(), FaceId{} );

    auto sortedIntersectionsIndices = sortIntersectionsAlongBaseEdge( mesh, baseEdge, edgeData, sortData );

    EdgeId e = baseEdge;       
    // disconnect edge e from its origin
    EdgeId e0;
    {
        EdgeId ePrev = mesh.topology.prev( e );
        if ( ePrev != e )
            mesh.topology.splice( ePrev, e );
        // e now becomes the second part of split edge, add first part to it
        e0 = mesh.topology.makeEdge();
        if ( ePrev != e )
            mesh.topology.splice( ePrev, e0 );
    }

    bool isAllLeftOnly = true;
    bool isAllRightOnly = true;
    for ( int i = 0; i < sortedIntersectionsIndices.size(); ++i )
    {
        const auto& vertEdge = edgeData.orgEdgeInLeftTri[sortedIntersectionsIndices[i]];
        const auto& interIndex = edgeData.intersections[sortedIntersectionsIndices[i]];
        const auto& inter = contours[interIndex.contourId].intersections[interIndex.intersectionId];

        bool isBaseSym = std::get<EdgeId>( inter.primitiveId ).sym() == baseEdge;

        EdgeId leftEdge, rightEdge;
        EdgeId& baseleft = isBaseSym ? rightEdge : leftEdge;
        EdgeId& baserigth = isBaseSym ? leftEdge : rightEdge;

        baseleft = vertEdge;
        baserigth = baseleft.valid() ? mesh.topology.next( baseleft ) : mesh.topology.edgePerVertex()[edgeData.newVerts[sortedIntersectionsIndices[i]]];
        if ( baseleft == baserigth )
            baserigth = EdgeId{};

        EdgeId lastEdge = e;
        if ( i + 1 < sortedIntersectionsIndices.size() )
            lastEdge = mesh.topology.makeEdge();

        if ( isAllLeftOnly && rightEdge.valid() )
            isAllLeftOnly = false;

        if ( isAllRightOnly && leftEdge.valid() )
            isAllRightOnly = false;

        connectEdges( mesh.topology, e0, lastEdge, leftEdge, rightEdge );

        e0 = lastEdge;
    }

    // fix triangle if this was last or first
    if ( isAllLeftOnly && oldRight.valid() )
        triangulateContour( mesh, e0.sym(), oldRight, new2OldMap );
    if ( isAllRightOnly && oldLeft.valid() )
        triangulateContour( mesh, e0, oldLeft, new2OldMap );
}

// this function cut mesh edge and connects it with result path, 
// after it each path edge left and right faces are invalid (they are removed)
void cutEdgesIntoPieces( Mesh& mesh, 
                         const Vector<EdgeData, UndirectedEdgeId>& edgeData, const OneMeshContours& contours, 
                         const SortIntersectionsData* sortData,
                         FaceMap* new2OldMap )
{
    MR_TIMER;
    for ( const auto& edgeInfo : edgeData )
    {
        if ( edgeInfo.intersections.empty() )
            continue;

        cutOneEdge( mesh, edgeInfo, contours, sortData, new2OldMap );
    }
}

void prepareFacesMap( const MeshTopology& topology, FaceMap& new2OldMap )
{
    new2OldMap.resize( topology.lastValidFace() + 1 );
    for ( auto f : topology.getValidFaces() )
        new2OldMap[f] = f;
}

// Checks if cut mesh has valid loops in intersections
// if error occurs returns bad faces bit set, otherwise returns empty one
FaceBitSet getBadFacesAfterCut( const MeshTopology& topology, const PreCutResult& preRes,
                               const FullRemovedFacesInfo& oldFaces )
{
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

CutMeshResult cutMesh( Mesh& mesh, const OneMeshContours& contours, const CutMeshParameters& params )
{
    MR_TIMER;
    MR_MESH_WRITER( mesh );

    CutMeshResult res;

    if ( params.new2OldMap )
        prepareFacesMap( mesh.topology, *params.new2OldMap );

    auto preRes = doPreCutMesh( mesh, contours );
    cutEdgesIntoPieces( mesh, preRes.edgeData, contours, params.sortData, params.new2OldMap );
    fixOrphans( mesh, preRes.paths, preRes.removedFaces, params.new2OldMap );

    res.fbsWithCountourIntersections = getBadFacesAfterCut( mesh.topology, preRes, preRes.removedFaces );
    if ( !params.forceFillAfterBadCut && res.fbsWithCountourIntersections.count() > 0 )
        return res;
    // fill contours
    for ( int pathId = 0; pathId < preRes.paths.size(); ++pathId )
    {
        const auto& path = preRes.paths[pathId];
        const auto& oldEdgesInfo = preRes.oldEdgesInfo[pathId];
        for ( int edgeId = 0; edgeId < path.size(); ++edgeId )
        {
            FaceId oldf = preRes.removedFaces[pathId][edgeId].f;
            if ( !oldf.valid() || res.fbsWithCountourIntersections.test( oldf ) )
                continue;
            if ( oldEdgesInfo[edgeId].hasLeft && !mesh.topology.left( path[edgeId] ).valid() )
                triangulateContour( mesh, path[edgeId], oldf, params.new2OldMap );
            if ( oldEdgesInfo[edgeId].hasRight && !mesh.topology.right( path[edgeId] ).valid() )
                triangulateContour( mesh, path[edgeId].sym(), oldf, params.new2OldMap );
        }
    }
    res.resultCut = std::move( preRes.paths );

    return res;
}

std::vector<MR::EdgePath> cutMeshWithPlane( MR::Mesh& mesh, const MR::Plane3f& plane, MR::FaceMap* mapNew2Old /*= nullptr*/ )
{
    MR_TIMER;
    MR_MESH_WRITER( mesh );

    auto sections = extractPlaneSections( mesh, -plane );
    auto contours = convertSurfacePathsToMeshContours( mesh, sections );
    CutMeshParameters params = {};
    params.new2OldMap = mapNew2Old;
    auto cutEdges = cutMesh( mesh, contours, params );
    
    FaceBitSet goodFaces = fillContourLeft( mesh.topology, cutEdges.resultCut );
    auto components = MeshComponents::getAllComponents( mesh, MeshComponents::FaceIncidence::PerVertex );
    for ( const auto& comp : components )
    {
        if ( ( comp & goodFaces ).any() )
            continue;
        // separated component
        auto point = mesh.orgPnt( mesh.topology.edgePerFace()[comp.find_first()] );
        if ( plane.distance( point ) >= 0.0f )
            goodFaces |= comp;
    }
    auto removedFaces = mesh.topology.getValidFaces() - goodFaces;

    mesh.topology.deleteFaces( removedFaces );
    if ( mapNew2Old )
    {
        MR::FaceMap& map = *mapNew2Old;
        for ( auto& faceId : removedFaces )
            map[faceId] = FaceId();
    }
    return cutEdges.resultCut;
}

TEST( MRMesh, BooleanIntersectionsSort )
{
    Mesh meshA;
    meshA.points = std::vector<Vector3f>
    {
        { 8.95297337f, 14.3548975f,-0.212119192f },
        { 8.98828983f, 14.3914976f,-0.198161319f },
        { 8.92162418f, 14.4169340f,-0.203402281f },
        { 8.95297337f, 14.4501600f,-0.191835344f }
    };
    std::vector<MeshBuilder::Triangle> trisA = 
    {
        { 0_v, 1_v, 3_v, 0_f },
        { 0_v, 3_v, 2_v, 1_f }
    };
    meshA.topology = MeshBuilder::fromTriangles( trisA );

    Mesh meshB;
    meshB.points = std::vector<Vector3f>
    {
        { 8.91892719f, 14.3419390f, -0.208497435f },
        { 8.99423218f, 14.4023476f, -0.208966389f },
        { 9.00031281f, 14.4126110f, -0.209267750f },
        { 8.99934673f, 14.4161797f, -0.209171638f },
        { 8.91623878f, 14.3510427f, -0.205425277f }
    };
    std::vector<MeshBuilder::Triangle> trisB =
    {
        { 0_v, 1_v, 2_v, 0_f },
        { 0_v, 2_v, 4_v, 1_f },
        { 2_v, 3_v, 4_v, 2_f }
    };
    meshB.topology = MeshBuilder::fromTriangles( trisB );
    auto converters = getVectorConverters( meshA, meshB );
    auto intersections = findCollidingEdgeTrisPrecise( meshA, meshB, converters.toInt );
    auto contours = orderIntersectionContours( meshA.topology, meshB.topology, intersections );
    auto meshAContours = getOneMeshIntersectionContours( meshA, meshB, contours, true, converters );
    auto meshBContours = getOneMeshIntersectionContours( meshA, meshB, contours, false, converters );

    SortIntersectionsData dataForA{meshB,contours,converters.toInt,nullptr,meshA.topology.vertSize(),false};
    
    Vector3f aNorm;
    for ( auto f : meshA.topology.getValidFaces() )
        aNorm += meshA.dirDblArea( f );
    aNorm = aNorm.normalized();
    CutMeshParameters params;
    params.sortData = &dataForA;
    cutMesh( meshA, meshAContours, params );

    for ( auto f : meshA.topology.getValidFaces() )
        EXPECT_TRUE( dot( meshA.dirDblArea( f ), aNorm ) > 0.0f );
}

} //namespace MR
