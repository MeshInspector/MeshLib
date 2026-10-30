#include "MROneMeshContours.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRParallelFor.h"
#include "MRAffineXf3.h"
#include "MRRingIterator.h"
#include "MRBox.h"
#include "MRSurfacePath.h"
#include "MRPch/MRSpdlog.h"

namespace MR
{

namespace
{

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
        if ( prev.isFace() )
        {
            assert( prev.face() == mesh.topology.left( curr.e ) );
        }
        if ( next.isFace() )
        {
            assert( next.face() == mesh.topology.left( curr.e ) );
        }
#endif
        return OneMeshIntersection{ mesh.topology.left( curr.e ),mesh.triPoint( curr ) };
    }
    else
    {
        auto v = curr.inVertex( mesh.topology );
        if ( v ) // curr in vertex
        {
            if ( prev.isVert() )
            {
                auto pV = prev.vert();
                if ( pV == v )
                    return {};
            }
            if ( next.isVert() )
            {
                auto nV = next.vert();
                if ( nV == v )
                    return {};
            }
            return OneMeshIntersection{ v,mesh.points[v] };
        }

        auto e = optMeshEdgePoint.e;

        // only need that prev and next are on different sides of curr
        if ( prev.isFace() )
        {
            auto pFId = prev.face();
            if ( mesh.topology.right( e ) != pFId )
            {
                e = e.sym(); // orient correct to be consistent with path
                assert( mesh.topology.right( e ) == pFId );
            }

            // without a face left of `e` (mesh boundary) the contour only touches `e` and returns to `prev` face;
            // at an ear tip (boundary vertex with one face) next( e ) is an edge of `prev` face, so check the face too
            if ( next.isVert() )
            {
                auto nVId = next.vert();
                if ( !mesh.topology.left( e ) || mesh.topology.dest( mesh.topology.next( e ) ) != nVId )
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
                    return OneMeshIntersection{ e,mesh.edgePoint( optMeshEdgePoint ) };
                }
            }
            else if ( next.isEdge() )
            {
                auto nEUId = next.edge().undirected();
                if ( !mesh.topology.left( e ) || ( mesh.topology.next( e ).undirected() != nEUId &&
                     mesh.topology.prev( e.sym() ).undirected() != nEUId ) )
                {
                    assert( nEUId == e.undirected() || nEUId == mesh.topology.prev( e ).undirected() || nEUId == mesh.topology.next( e.sym() ).undirected() );
                    return {};
                }
                else
                {
                    return OneMeshIntersection{ e,mesh.edgePoint( optMeshEdgePoint ) };
                }
            }
            else // if face
            {
                auto nFId = next.face();
                if ( pFId == nFId )
                {
                    return {};
                }
                else
                {
                    assert( nFId == mesh.topology.left( e ) );
                    return OneMeshIntersection{ e,mesh.edgePoint( optMeshEdgePoint ) };
                }
            }
        }
        else // if next == face and prev is not face, symetric
        {
            auto nFId = next.face();
            if ( mesh.topology.left( e ) != nFId )
            {
                e = e.sym();
                assert( mesh.topology.left( e ) == nFId );
            }

            // the same without a face right of `e`
            if ( prev.isVert() )
            {
                auto pVId = prev.vert();
                if ( !mesh.topology.right( e ) || mesh.topology.dest( mesh.topology.prev( e ) ) != pVId )
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
                    return OneMeshIntersection{ e,mesh.edgePoint( optMeshEdgePoint ) };
                }

            }
            else // if edge
            {
                auto pEUId = prev.edge().undirected();
                if ( !mesh.topology.right( e ) || ( mesh.topology.prev( e ).undirected() != pEUId &&
                     mesh.topology.next(e.sym() ).undirected() != pEUId ) )
                {
                    assert( pEUId == e.undirected() || pEUId == mesh.topology.next( e ).undirected() || pEUId == mesh.topology.prev( e.sym() ).undirected() );
                    return {};
                }
                else
                {
                    return OneMeshIntersection{ e,mesh.edgePoint( optMeshEdgePoint ) };
                }
            }
        }
    }
}

FaceId findSharedFaceImpl( const MeshTopology& topology, VertId vid, EdgeId eid, const MeshTriPoint& mtp )
{
    auto mtpEdgeRep = mtp.onEdge( topology );
    if ( !mtpEdgeRep )
        return topology.left( mtp.e );

    if ( topology.dest( eid ) == vid )
        eid = eid.sym();

    auto mtpVertRep = mtp.inVertex( topology );
    if ( mtpVertRep.valid() )
    {
        if ( topology.dest( topology.next( eid ) ) == mtpVertRep )
            return topology.left( eid );
        else if ( topology.dest( topology.prev( eid ) ) == mtpVertRep )
            return topology.right( eid );
        else if ( topology.org( eid ) == mtpVertRep || topology.dest( eid ) == mtpVertRep )
            return topology.left( eid ); // mtp is an end of eid itself (the contour turns back in this vertex), so any face of eid is shared
    }
    else
    {
        auto mtpEUndir = mtpEdgeRep.e.undirected();
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

// finds FaceId if face, shared among vid eid and mtp, if there is no one returns left(mtp.e);
// if the found face does not exist (mtp on the mesh boundary), returns any existing face of eid
FaceId findSharedFace( const MeshTopology& topology, VertId vid, EdgeId eid, const MeshTriPoint& mtp )
{
    if ( auto f = findSharedFaceImpl( topology, vid, eid, mtp ) )
        return f;
    return topology.left( eid ) ? topology.left( eid ) : topology.right( eid );
}

// true if v is the third vertex of an existing face of e; across the hole of a boundary edge there is no face to share
bool isFaceApex( const MeshTopology& topology, VertId v, EdgeId e )
{
    return ( topology.left( e ) && topology.dest( topology.next( e ) ) == v ) || ( topology.right( e ) && topology.dest( topology.prev( e ) ) == v );
}

// function to get information about mesh tri point intersection between two surfaces paths
// and process it correctly
std::optional<OneMeshIntersection> centralIntersection( const Mesh& mesh, const OneMeshIntersection& prev, const MeshTriPoint& curr, const OneMeshIntersection& next,
                                                float closeEdgeEps,
                                                CenterInterType& type )
{
    MR_TIMER;
    type = CenterInterType::Common;
    if ( prev.isFace() || next.isFace() )
        return centralIntersectionForFaces( mesh, prev, curr, next );

    const auto& topology = mesh.topology;
    if ( prev.isVert() )
    {
        auto pVId = prev.vert();
        if ( next.isVert() )
        {
            auto nVId = next.vert();
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
        else if ( next.isEdge() )
        {
            auto nEId = next.edge();
            if ( isFaceApex( topology, pVId, nEId ) ) // in any orientation
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
        auto vid = curr.inVertex( topology );
        if ( vid.valid() )
            return OneMeshIntersection{vid,mesh.points[vid]};
        if ( topology.dest( topology.prev( edgeOp.e ) ) == pVId )
            return OneMeshIntersection{edgeOp.e,mesh.edgePoint( edgeOp )};
        else
            return OneMeshIntersection{edgeOp.e.sym(),mesh.edgePoint( edgeOp )};
    }
    else if ( prev.isEdge() )
    {
        auto pEId = prev.edge();
        if ( next.isVert() )
        {
            auto nVId = next.vert();
            // `next` is the third vertex of the left face of `prev` edge; if there is no left face (a boundary edge), only an ear tip
            // still has `next` as the third vertex of its right face too
            if ( topology.dest( topology.next( pEId ) ) == nVId && ( topology.left( pEId ) || topology.dest( topology.prev( pEId ) ) == nVId ) )
                return {};
            if ( topology.dest( pEId ) == nVId || topology.org( pEId ) == nVId || ( topology.dest( topology.prev( pEId ) ) == nVId && topology.right( pEId ) ) )
            {
                assert( fromSameTriangle( topology, mesh.toTriPoint( nVId ), MeshTriPoint( MeshEdgePoint( pEId, 0.5f ) ) ) );
                FaceId sharedFaceWithCur = findSharedFace( topology, nVId, pEId, curr );
                auto verts = topology.getTriVerts( sharedFaceWithCur );
                if ( verts[0] == nVId || verts[1] == nVId || verts[2] == nVId )
                    return OneMeshIntersection{ sharedFaceWithCur,mesh.triPoint( curr ) };
                // we fall here if prev and next have shared face, but this face is not shared with curr,
                // if this happens we continue to default case
            }
        }
        else if ( next.isEdge() )
        {
            auto nEId = next.edge();
            if ( nEId.undirected() == pEId.undirected() )
            {
                FaceId currF = findSharedFace( mesh.topology, mesh.topology.dest( nEId ), nEId, curr );
                assert( currF );
                auto coordDif = ( next.coordinate - prev.coordinate ).length();
                if ( coordDif < closeEdgeEps )
                {
                    type = CenterInterType::SameEdgesClosePos;
                    if ( currF == topology.left( nEId ) && topology.right( nEId ) )
                        currF = topology.right( nEId );
                    else if ( currF == topology.right( nEId ) && topology.left( nEId ) )
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
            if ( topology.left( pEId ) && ( topology.next( pEId ).undirected() == nEId.undirected() ||
                 topology.prev( pEId.sym() ).undirected() == nEId.sym().undirected() ) )
            {
                // orientation is consistent or `next` orientation can be wrong
                auto edgeOp = curr.onEdge( topology );
                if ( edgeOp )
                    return {};
                return OneMeshIntersection{topology.left( curr.e ),mesh.triPoint( curr )};
            }
            else if ( topology.right( pEId ) && ( topology.prev( pEId ).undirected() == nEId.undirected() ||
                      topology.next( pEId.sym() ).undirected() == nEId.sym().undirected() ) )
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
        auto vid = curr.inVertex( topology );
        if ( vid.valid() )
            return OneMeshIntersection{vid,mesh.points[vid]};
        // .undirected() here since `pEId` might be non-consistent on first intersection
        if ( topology.prev( edgeOp.e ).undirected() == pEId.undirected() || topology.next( edgeOp.e.sym() ).undirected() == pEId.sym().undirected() )
            return OneMeshIntersection{edgeOp.e,mesh.edgePoint( edgeOp )};
        else
            return OneMeshIntersection{edgeOp.e.sym(),mesh.edgePoint( edgeOp )};
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
        auto v = mtp.inVertex( mesh.topology );
        if ( v )
            res.setPrimitive( v );
        else
            res.setPrimitive( e.e );
    }
    else
        res.setPrimitive( mesh.topology.left( mtp.e ) );
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

} //anonymous namespace

void subdivideLoneContours( Mesh& mesh, const OneMeshContours& contours, FaceHashMap* new2oldMap /*= nullptr */ )
{
    MR_TIMER;
    HashMap<FaceId, Vector<int>> face2contoursMap;
    for ( int i = 0; i < contours.size(); ++i )
    {
        FaceId f = contours[i].intersections.front().face();
        face2contoursMap[f].push_back( i );
    }
    FaceHashMap splitNew2Old; // new face -> face it was split from here, to update AABB tree instead of rebuilding it
    for ( auto& [faceId, conts] : face2contoursMap )
    {
        assert( !conts.empty() );
        Vector3f massCenter;
        int counter = 0;
        // here we find centroid of first lone contour of this face because
        // splitting with average centroid does not guarantee that any lone contour will be subdivided
        // and splitting with first contour centroid guarantee that at least this contour will be subdivided
        for ( const auto& p : contours[conts.front()].intersections )
        {
            ++counter;
            massCenter += p.coordinate;
        }
        massCenter /= float( counter );
        const FaceId firstNewFace( mesh.topology.faceSize() );
        mesh.splitFace( faceId, massCenter, nullptr, new2oldMap );
        for ( FaceId f = firstNewFace; f < mesh.topology.faceSize(); ++f )
            splitNew2Old[f] = faceId;
    }
    mesh.updateCachesAfterSplits( splitNew2Old );
}

void getOneMeshIntersectionContours( const Mesh& meshA, const Mesh& meshB, const ContinuousContours& contours,
    OneMeshContours* outA, OneMeshContours* outB,
    const CoordinateConverters& converters, const AffineXf3f* rigidB2A, Contours3f* outPtsA, bool addSelfyTerminalVerts )
{
    MR_TIMER;
    assert( outA || outB || outPtsA );
    // addSelfyTerminalVerts is supported only if both meshes are actually the same without any relative transformation
    assert( !addSelfyTerminalVerts || ( &meshA == &meshB && !rigidB2A ) );

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
    // the terminal vertices before and after the open self-intersection contours
    const auto& topologyA = meshA.topology;
    std::vector<std::array<VertId, 2>> terminals;
    if ( addSelfyTerminalVerts )
    {
        terminals.resize( contours.size() );
        for ( int j = 0; j < contours.size(); ++j )
        {
            const auto& c = contours[j];
            if ( c.empty() || isClosed( c ) )
                continue;
            if ( topologyA.right( c.front().edge ) )
            {
                terminals[j][0] = topologyA.dest( topologyA.prev( c.front().edge ) );
                assert( topologyA.isTriVert( c.front().tri(), terminals[j][0] ) );
            }
            if ( topologyA.left( c.back().edge ) )
            {
                terminals[j][1] = topologyA.dest( topologyA.next( c.back().edge ) );
                assert( topologyA.isTriVert( c.back().tri(), terminals[j][1] ) );
            }
        }
    }

    // contour j of the output has its intersections from outOffsets[j] + (a vertex before it), after the input ones from inOffsets[j];
    // only the sizes here, which are next to each other: reading the contours themselves on this thread costs a cache miss per contour
    std::vector<int> inOffsets( contours.size() + 1 ), outOffsets( contours.size() + 1 );
    for ( int j = 0; j < contours.size(); ++j )
    {
        const auto n = int( contours[j].size() );
        inOffsets[j + 1] = inOffsets[j] + n;
        outOffsets[j + 1] = outOffsets[j] + n + ( terminals.empty() ? 0 : int( terminals[j][0].valid() ) + int( terminals[j][1].valid() ) );
    }
    // resizeNoInit touches no element but still takes a step per element, so the outputs are sized concurrently
    tbb::task_group sizing;
    for ( auto out : { outA, outB } )
        if ( out )
            sizing.run( [out, &outOffsets] { out->intersections.resizeNoInit( outOffsets.back() ); } ); // every element is written below
    if ( outPtsA )
    {
        outPtsA->resize( contours.size() );
        for ( int j = 0; j < contours.size(); ++j )
            resizeNoInit( ( *outPtsA )[j], outOffsets[j + 1] - outOffsets[j] );
    }
    sizing.wait();

    BitSet closed( contours.size() );
    parallelForContourPieces( inOffsets, [&] ( int j, int k0, int k1 )
    {
        if ( k0 == inOffsets[j] && isClosed( contours[j] ) ) // the piece with the first intersection of the contour
            closed.atomicTestSet( j );
        const int shift = outOffsets[j] - inOffsets[j] + ( terminals.empty() ? 0 : int( terminals[j][0].valid() ) );
        for ( int k = k0; k < k1; ++k )
        {
            Vector3f a, b, c, d, e;
            const auto& inIntersection = contours[j][k - inOffsets[j]];
            OneMeshIntersection pntA, pntB;

            if ( inIntersection.isEdgeATriB() )
            {
                pntA.setPrimitive( inIntersection.edge );
                pntB.setPrimitive( inIntersection.tri() );
                meshB.getTriPoints( inIntersection.tri(), a, b, c );
                d = meshA.orgPnt( inIntersection.edge );
                e = meshA.destPnt( inIntersection.edge );
            }
            else
            {
                pntB.setPrimitive( inIntersection.edge );
                pntA.setPrimitive( inIntersection.tri() );
                meshA.getTriPoints( inIntersection.tri(), a, b, c );
                d = meshB.orgPnt( inIntersection.edge );
                e = meshB.destPnt( inIntersection.edge );
            }
            // always calculate in mesh A space
            pntA.coordinate = findTriangleSegmentIntersectionPrecise(
                getCoord( a, !inIntersection.isEdgeATriB() ),
                getCoord( b, !inIntersection.isEdgeATriB() ),
                getCoord( c, !inIntersection.isEdgeATriB() ),
                getCoord( d, inIntersection.isEdgeATriB() ),
                getCoord( e, inIntersection.isEdgeATriB() ), converters );

            const int o = k + shift;
            if ( outA )
                outA->intersections[o] = pntA;
            if ( outPtsA )
                ( *outPtsA )[j][o - outOffsets[j]] = pntA.coordinate;
            if ( outB )
            {
                pntB.coordinate = rigidB2A ? inverseXf( pntA.coordinate ) : pntA.coordinate;
                outB->intersections[o] = pntB;
            }
        }
    } );

    for ( auto out : { outA, outB } )
    {
        if ( !out )
            continue;
        out->offsets = outOffsets;
        out->closed = closed;
    }

    for ( int j = 0; j < terminals.size(); ++j )
    {
        for ( int t = 0; t < 2; ++t )
        {
            const auto v = terminals[j][t];
            if ( !v )
                continue;
            const int o = t == 0 ? outOffsets[j] : outOffsets[j + 1] - 1;
            const OneMeshIntersection inter{ v, meshA.points[v] };
            if ( outA )
                outA->intersections[o] = inter;
            if ( outPtsA )
                ( *outPtsA )[j][o - outOffsets[j]] = inter.coordinate;
            if ( outB )
                outB->intersections[o] = inter;
        }
    }
}

OneMeshContours::OneMeshContours( const Vector<OneMeshContour>& contours )
{
    size_t n = 0;
    for ( const auto& c : contours )
        n += c.intersections.size();
    intersections.reserve( n );
    for ( const auto& c : contours )
        push_back( c );
}

OneMeshContour OneMeshContours::contour( size_t c ) const
{
    const auto v = ( *this )[c];
    OneMeshContour res{ .closed = v.closed };
    res.intersections.insert( res.intersections.end(), v.intersections.begin(), v.intersections.end() );
    return res;
}

void OneMeshContours::push_back( const OneMeshContourView& contour )
{
    if ( contour.closed )
        closed.autoResizeSet( size() );
    intersections.insert( intersections.end(), contour.intersections.begin(), contour.intersections.end() );
    offsets.push_back( int( intersections.size() ) );
}

void parallelForContourPieces( const std::vector<int>& offsets, const std::function<void( int c, int i0, int i1 )>& f )
{
    assert( !offsets.empty() && offsets.front() == 0 );
    tbb::parallel_for( tbb::blocked_range<int>( 0, offsets.back() ), [&] ( const tbb::blocked_range<int>& range )
    {
        // the last contour starting at range.begin() or before it contains it (empty contours there start at it too, but go before it)
        auto c = int( std::upper_bound( offsets.begin(), offsets.end(), range.begin() ) - offsets.begin() ) - 1;
        for ( int i0 = range.begin(); i0 < range.end(); ++c )
        {
            const int i1 = std::min( offsets[c + 1], range.end() );
            if ( i0 < i1 )
                f( c, i0, i1 );
            i0 = i1;
        }
    } );
}

OneMeshContours getOneMeshSelfIntersectionContours( const Mesh& mesh, const ContinuousContours& contours, const CoordinateConverters& converters, const AffineXf3f* rigidB2A /*= nullptr */ )
{
    MR_TIMER;
    OneMeshContours res;
    AffineXf3f inverseXf;
    if ( rigidB2A )
        inverseXf = rigidB2A->inverse();
    // the offsets are filled in a vector of their own: GCC sees resize() of the default { 0 } as out of its bounds
    std::vector<int> offsets( contours.size() + 1 );
    for ( int j = 0; j < contours.size(); ++j )
    {
        offsets[j + 1] = offsets[j] + int( contours[j].size() );
        if ( isClosed( contours[j] ) )
            res.closed.autoResizeSet( j );
    }
    res.offsets = std::move( offsets );
    res.intersections.resizeNoInit( res.offsets.back() );

    parallelForContourPieces( res.offsets, [&] ( int j, int k0, int k1 )
    {
        for ( int k = k0; k < k1; ++k )
        {
            const auto& inIntersection = contours[j][k - res.offsets[j]];
            auto& outIntersection = res.intersections[k];

            if ( !rigidB2A == inIntersection.isEdgeATriB() )
                outIntersection.setPrimitive( inIntersection.edge );
            else
                outIntersection.setPrimitive( inIntersection.tri() );

            Vector3f a, b, c, d, e;
            mesh.getTriPoints( inIntersection.tri(), a, b, c );
            d = mesh.orgPnt( inIntersection.edge );
            e = mesh.destPnt( inIntersection.edge );

            // always calculate in mesh A space
            outIntersection.coordinate = findTriangleSegmentIntersectionPrecise(
                rigidB2A ? ( *rigidB2A )( a ) : a,
                rigidB2A ? ( *rigidB2A )( b ) : b,
                rigidB2A ? ( *rigidB2A )( c ) : c,
                rigidB2A ? ( *rigidB2A )( d ) : d,
                rigidB2A ? ( *rigidB2A )( e ) : e, converters );

            if ( rigidB2A )
                outIntersection.coordinate = inverseXf( outIntersection.coordinate );
        }
    } );
    return res;
}

Contours3f extractMeshContours( const OneMeshContours& meshContours )
{
    Contours3f res( meshContours.size() );
    for ( int i = 0; i < res.size(); ++i )
    {
        auto& resI = res[i];
        const auto& imputI = meshContours[i].intersections;
        resI.resize( imputI.size() );
        for ( int j = 0; j < resI.size(); ++j )
            resI[j] = imputI[j].coordinate;
    }
    return res;
}

Expected<OneMeshContour> surfaceLineToMeshContour( const Mesh& mesh, const std::vector<MeshTriPoint>& meshTriPointsOrg,
    MeshTriPointsConnector connectorFn /*= {}*/, std::vector<int>* pivotIndices /*= nullptr */ )
{
    MR_TIMER;
    if ( meshTriPointsOrg.size() < 2 )
        return {};
    bool closed =
        meshTriPointsOrg.front().e == meshTriPointsOrg.back().e &&
        meshTriPointsOrg.front().bary.a == meshTriPointsOrg.back().bary.a &&
        meshTriPointsOrg.front().bary.b == meshTriPointsOrg.back().bary.b;

    if ( closed && meshTriPointsOrg.size() < 4 )
        return {};

    if ( pivotIndices )
        pivotIndices->resize( meshTriPointsOrg.size(), -1 );
    // clear duplicates
    auto meshTriPoints = meshTriPointsOrg;
    if ( closed )
        meshTriPoints.resize( meshTriPoints.size() - 1 );
    size_t sizeMTP = closed ? meshTriPoints.size() : meshTriPoints.size() - 1;

    Vector<int> sameEdgeMTPs;
    Box3f box;
    for ( int i = 0; i < sizeMTP; ++i )
    {
        box.include( mesh.triPoint( meshTriPoints[i] ) );
        const auto& mtp1 = meshTriPoints[i];
        const auto& mtp2 = meshTriPoints[( i + 1 ) % meshTriPoints.size()];
        auto e1 = mtp1.onEdge( mesh.topology );
        auto e2 = mtp2.onEdge( mesh.topology );
        if ( !e1 || !e2 )
            continue;
        auto v1 = mtp1.inVertex( mesh.topology );
        auto v2 = mtp2.inVertex( mesh.topology );
        if ( v1.valid() && v2.valid() )
        {
            if ( v1 == v2 )
                sameEdgeMTPs.push_back( i );
        }
        else
        {
            if ( e1.e.undirected() == e2.e.undirected() )
                sameEdgeMTPs.push_back( i );
        }
    }
    for ( int i = int( sameEdgeMTPs.size() ) - 1; i >= 0; --i )
        meshTriPoints.erase( meshTriPoints.begin() + sameEdgeMTPs[i] );

    // build paths
    if ( meshTriPoints.size() < 2 )
        return {};

    if ( !connectorFn )
    {
        connectorFn = [&] ( const MeshTriPoint& start, const MeshTriPoint& end, int, int )->Expected<SurfacePath>
        {
            auto res = computeGeodesicPath( mesh, start, end );
            if ( !res.has_value() )
                return unexpected( toString( res.error() ) );
            return *res;
        };
    }

    int sameMtpsNavigator = 0;
    int pivotNavigator = 0;
    sizeMTP = closed ? meshTriPoints.size() : meshTriPoints.size() - 1;

    OneMeshContour res;
    Vector<OneMeshContour> surfacePaths( sizeMTP );
    for ( int i = 0; i < sizeMTP; ++i )
    {
        while ( sameMtpsNavigator < sameEdgeMTPs.size() && pivotNavigator == sameEdgeMTPs[sameMtpsNavigator] )
        {
            ++pivotNavigator;
            ++sameMtpsNavigator;
        }
        int firstIndex = pivotNavigator;
        ++pivotNavigator;
        int secondIndex = pivotNavigator;
        int secSameNav = sameMtpsNavigator;
        while ( secSameNav < sameEdgeMTPs.size() && secondIndex == sameEdgeMTPs[secSameNav] )
        {
            ++secondIndex;
            ++secSameNav;
            if ( secondIndex >= meshTriPointsOrg.size() )
            {
                secondIndex = 0;
                secSameNav = 0;
            }
        }
        // using DijkstraAStar here might be faster, in most case points are close to each other
        auto sp = connectorFn( meshTriPoints[i], meshTriPoints[( i + 1 ) % meshTriPoints.size()], firstIndex, secondIndex );
        if ( !sp.has_value() )
            return unexpected( sp.error() );
        auto partContours = surfacePathsToMeshContours( mesh, { std::move( sp.value() ) } );
        assert( partContours.size() == 1 );
        surfacePaths[i] = partContours.contour( 0 );
        if ( surfacePaths[i].intersections.size() == 1 )
        {
            // if lone make sure that prev MeshTriPoint is on the right (if on the left - sym)
            if ( surfacePaths[i].intersections[0].isEdge() )
            {
                auto& inter = surfacePaths[i].intersections[0];
                const auto edge = inter.edge();
                auto onEdge = meshTriPoints[i].onEdge( mesh.topology );
                if ( !onEdge )
                {
                    // if mtp is on face - make sure that it is right of edge
                    if ( mesh.topology.left( edge ) == mesh.topology.left( meshTriPoints[i].e ) )
                        inter.setPrimitive( edge.sym() );
                }
                else
                {
                    auto inVert = meshTriPoints[i].inVertex( mesh.topology );
                    if ( !inVert )
                    {
                        // if mtp is on edge - make sure it is prev(e) or next(e.sym)
                        if ( mesh.topology.next( edge ).undirected() == onEdge.e.undirected() ||
                            mesh.topology.prev( edge.sym() ).undirected() == onEdge.e.undirected() )
                            inter.setPrimitive( edge.sym() );
                    }
                    else
                    {
                        // if mtp is in vert - make sure it is dest(prev(e))
                        if ( inVert == mesh.topology.dest( mesh.topology.next( edge ) ) )
                            inter.setPrimitive( edge.sym() );
                    }
                }
            }
        }
    }

    const int numMtps = int( meshTriPoints.size() );
    Vector<int> realPivots; // index in meshTriPointsOrg of every meshTriPoints[i]
    if ( pivotIndices )
    {
        realPivots.resize( numMtps );
        sameMtpsNavigator = 0;
        pivotNavigator = 0;
        for ( auto& realPivot : realPivots )
        {
            while ( sameMtpsNavigator < sameEdgeMTPs.size() && pivotNavigator == sameEdgeMTPs[sameMtpsNavigator] )
            {
                ++pivotNavigator;
                ++sameMtpsNavigator;
            }
            realPivot = pivotNavigator++;
        }
    }

    // a closed contour is walked from a point with non-empty incoming and outgoing paths, so that the previous
    // intersection of every point is already known when the point is processed (not a stand-in from an empty path)
    int startI = 0;
    if ( closed )
    {
        int startScore = -1;
        for ( int i = 0; i < numMtps && startScore < 2; ++i )
        {
            const int score = int( !surfacePaths[( i + numMtps - 1 ) % numMtps].intersections.empty() ) * ( 1 + int( !surfacePaths[i].intersections.empty() ) );
            if ( score > startScore )
            {
                startScore = score;
                startI = i;
            }
        }
    }

    const float closeEdgeEps = std::numeric_limits<float>::epsilon() * box.diagonal();
    int firstMtpStart = 0; // position in res where the intersections of meshTriPoints[0] start
    // add interjacent
    for ( int k = 0; k < numMtps; ++k )
    {
        const int i = ( startI + k ) % numMtps;
        if ( i == 0 )
            firstMtpStart = int( res.intersections.size() );
        const int realPivotIndex = pivotIndices ? realPivots[i] : -1;

        Vector<OneMeshIntersection>* prevInter = ( closed || i > 0 ) ? &surfacePaths[( i + int( meshTriPoints.size() ) - 1 ) % meshTriPoints.size()].intersections : nullptr;
        const Vector<OneMeshIntersection>* nextInter = ( i < sizeMTP ) ? &surfacePaths[i].intersections : nullptr;
        OneMeshIntersection lastPrev;
        OneMeshIntersection firstNext;
        if ( prevInter )
        {
            if ( prevInter->empty() )
            {
                if ( !res.intersections.empty() )
                    lastPrev = res.intersections.back();
                else
                    lastPrev = intersectionFromMeshTriPoint( mesh, meshTriPoints[( i + int( meshTriPoints.size() ) - 1 ) % meshTriPoints.size()] );
            }
            else
                lastPrev = prevInter->back();
        }
        else
            lastPrev = intersectionFromMeshTriPoint( mesh, meshTriPoints[i] );
        if ( nextInter )
        {
            if ( nextInter->empty() )
            {
                firstNext = intersectionFromMeshTriPoint( mesh, meshTriPoints[( i + 1 ) % meshTriPoints.size()] );
            }
            else
                firstNext = nextInter->front();
        }
        else
            firstNext = intersectionFromMeshTriPoint( mesh, meshTriPoints[i] );

        CenterInterType type;
        auto centerInterOp = centralIntersection( mesh, lastPrev, meshTriPoints[i], firstNext, closeEdgeEps, type );
        if ( centerInterOp )
        {
            bool mtpPushed = false;
            if ( type != CenterInterType::SameEdgesClosePos )
            {
                res.intersections.push_back( *centerInterOp );
                mtpPushed = true;
            }
            else
            {
                if ( res.intersections.empty() )
                {
                    if ( prevInter && !prevInter->empty() )
                        prevInter->back() = *centerInterOp;
                }
                else
                {
                    res.intersections.back() = *centerInterOp;
                    mtpPushed = true;
                    if ( i == 0 )
                        --firstMtpStart; // the first point took over the last intersection of the previous point
                }
            }
            if ( pivotIndices && mtpPushed )
            {
                int currentIndex = int( res.intersections.size() ) - 1;
                const int prevPivotIndex = ( closed || i > 0 ) ? realPivots[( i + numMtps - 1 ) % numMtps] : -1;
                if ( prevPivotIndex >= 0 && ( *pivotIndices )[prevPivotIndex] == currentIndex )
                    ( *pivotIndices )[prevPivotIndex] = -1;
                ( *pivotIndices )[realPivotIndex] = currentIndex;
            }
        }
        if ( nextInter && !nextInter->empty() )
        {
            if ( type == CenterInterType::Common )
                res.intersections.insert( res.intersections.end(), nextInter->begin(), nextInter->end() );
            else
                res.intersections.insert( res.intersections.end(), nextInter->begin() + 1, nextInter->end() );
        }
    }
    if ( firstMtpStart > 0 )
    {
        // restore the order starting from the first point
        const int numInters = int( res.intersections.size() );
        std::rotate( res.intersections.begin(), res.intersections.begin() + firstMtpStart, res.intersections.end() );
        if ( pivotIndices )
            for ( auto& p : *pivotIndices )
                if ( p >= 0 )
                    p = ( p - firstMtpStart + numInters ) % numInters;
    }
    if ( closed && !res.intersections.empty() )
    {
        res.intersections.push_back( res.intersections.front() );
        res.closed = true;
        if ( pivotIndices )
            ( *pivotIndices ).back() = ( *pivotIndices ).front();
    }
    return res;
}

Expected<OneMeshContour> surfaceLineToMeshContour( const Mesh& mesh, const std::vector<MeshTriPoint>& meshTriPointsOrg,
    SearchPathSettings searchSettings, std::vector<int>* pivotIndices )
{
    MeshTriPointsConnector conFn = [&] ( const MeshTriPoint& start, const MeshTriPoint& end, int, int )->Expected<SurfacePath>
    {
        auto res = computeGeodesicPath( mesh, start, end, searchSettings.geodesicPathApprox, searchSettings.maxReduceIters );
        if ( !res.has_value() )
            return unexpected( toString( res.error() ) );
        return *res;
    };
    return surfaceLineToMeshContour( mesh, meshTriPointsOrg, conFn, pivotIndices );
}

Expected<OneMeshContour> surfaceLineToClosedMeshContour( const Mesh& mesh, const std::vector<MeshTriPoint>& meshTriPointsOrg,
    SearchPathSettings searchSettings, std::vector<int>* pivotIndices )
{
    auto conts = meshTriPointsOrg;
    conts.push_back( meshTriPointsOrg.front() );
    return surfaceLineToMeshContour( mesh, conts, searchSettings, pivotIndices );
}

OneMeshContour surfacePathWithEndsToMeshContour( const MR::Mesh& mesh, const MeshTriPoint& start, const MR::SurfacePath& surfacePath, const MeshTriPoint& end )
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
            updatedPath.front() = startMep;
        for ( int i = 0; i < surfacePath.size(); ++i )
            updatedPath[i + shift] = surfacePath[i];
        if ( endMep )
            updatedPath.back() = endMep;
        res = surfacePathsToMeshContours( mesh, { updatedPath } ).contour( 0 );
    }
    else
    {
        res = surfacePathsToMeshContours( mesh, { surfacePath } ).contour( 0 );
    }
    if ( !startMep )
        res.intersections.insert( res.intersections.begin(), intersectionFromMeshTriPoint( mesh, start ) );
    if ( !endMep )
        res.intersections.push_back( intersectionFromMeshTriPoint( mesh, end ) );

    if ( res.intersections.front().samePrimitive( res.intersections.back() ) &&
         res.intersections.front().coordinate == res.intersections.back().coordinate )
        res.closed = true;

    return res;
}

OneMeshContours surfacePathsToMeshContours( const Mesh& mesh, const std::vector<SurfacePath>& surfacePaths )
{
    MR_TIMER;
    OneMeshContours res;
    // the offsets are filled in a vector of their own: GCC sees push_back() to the default { 0 } as out of its bounds
    std::vector<int> offsets( surfacePaths.size() + 1 );

    for ( int j = 0; j < surfacePaths.size(); ++j )
    {
        const auto curInContour = formatSurfacePath( mesh.topology, surfacePaths[j] );
        if ( curInContour.size() > 1 )
        {
            const auto& front = curInContour.front();
            const auto& back = curInContour.back();
            auto vF = front.inVertex( mesh.topology );
            auto vB = back.inVertex( mesh.topology );
            if ( ( vF.valid() && vF == vB ) || ( !vF && !vB && front.e == back.e && front.a == back.a ) )
                res.closed.autoResizeSet( j );
        }

        const auto start = res.intersections.size();
        res.intersections.resize( start + curInContour.size() );
        offsets[j + 1] = int( res.intersections.size() );
        ParallelFor( curInContour, [&] ( size_t i )
        {
            const auto& inIntersection = curInContour[i];
            auto& outIntersection = res.intersections[start + i];
            const auto vid = inIntersection.inVertex( mesh.topology );
            if ( vid.valid() )
                outIntersection.setPrimitive( vid );
            else
                outIntersection.setPrimitive( inIntersection.e );
            outIntersection.coordinate = mesh.edgePoint( inIntersection );
        } );
    }
    res.offsets = std::move( offsets );
    return res;
}

} //namespace MR
