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
        if ( std::holds_alternative<FaceId>( prev.primitiveId ) )
        {
            assert( std::get<FaceId>( prev.primitiveId ) == mesh.topology.left( curr.e ) );
        }
        if ( std::holds_alternative<FaceId>( next.primitiveId ) )
        {
            assert( std::get<FaceId>( next.primitiveId ) == mesh.topology.left( curr.e ) );
        }
#endif
        return OneMeshIntersection{ mesh.topology.left( curr.e ),mesh.triPoint( curr ) };
    }
    else
    {
        auto v = curr.inVertex( mesh.topology );
        if ( v ) // curr in vertex
        {
            if ( std::holds_alternative<VertId>( prev.primitiveId ) )
            {
                auto pV = std::get<VertId>( prev.primitiveId );
                if ( pV == v )
                    return {};
            }
            if ( std::holds_alternative<VertId>( next.primitiveId ) )
            {
                auto nV = std::get<VertId>( next.primitiveId );
                if ( nV == v )
                    return {};
            }
            return OneMeshIntersection{ v,mesh.points[v] };
        }

        auto e = optMeshEdgePoint.e;

        // only need that prev and next are on different sides of curr
        if ( std::holds_alternative<FaceId>( prev.primitiveId ) )
        {
            auto pFId = std::get<FaceId>( prev.primitiveId );
            if ( mesh.topology.right( e ) != pFId )
            {
                e = e.sym(); // orient correct to be consistent with path
                assert( mesh.topology.right( e ) == pFId );
            }

            if ( std::holds_alternative<VertId>( next.primitiveId ) )
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
                    return OneMeshIntersection{ e,mesh.edgePoint( optMeshEdgePoint ) };
                }
            }
            else if ( std::holds_alternative<EdgeId>( next.primitiveId ) )
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
                    return OneMeshIntersection{ e,mesh.edgePoint( optMeshEdgePoint ) };
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
                    return OneMeshIntersection{ e,mesh.edgePoint( optMeshEdgePoint ) };
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

            if ( std::holds_alternative<VertId>( prev.primitiveId ) )
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
                    return OneMeshIntersection{ e,mesh.edgePoint( optMeshEdgePoint ) };
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
                    return OneMeshIntersection{ e,mesh.edgePoint( optMeshEdgePoint ) };
                }
            }
        }
    }
}

// finds FaceId if face, shared among vid eid and mtp, if there is no one returns left(mtp.e)
FaceId findSharedFace( const MeshTopology& topology, VertId vid, EdgeId eid, const MeshTriPoint& mtp )
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

// function to get information about mesh tri point intersection between two surfaces paths
// and process it correctly
std::optional<OneMeshIntersection> centralIntersection( const Mesh& mesh, const OneMeshIntersection& prev, const MeshTriPoint& curr, const OneMeshIntersection& next,
                                                float closeEdgeEps,
                                                CenterInterType& type )
{
    MR_TIMER;
    type = CenterInterType::Common;
    if ( std::holds_alternative<FaceId>( prev.primitiveId ) || std::holds_alternative<FaceId>( next.primitiveId ) )
        return centralIntersectionForFaces( mesh, prev, curr, next );

    const auto& topology = mesh.topology;
    if ( std::holds_alternative<VertId>( prev.primitiveId ) )
    {
        auto pVId = std::get<VertId>( prev.primitiveId );
        if ( std::holds_alternative<VertId>( next.primitiveId ) )
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
        else if ( std::holds_alternative<EdgeId>( next.primitiveId ) )
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
        auto vid = curr.inVertex( topology );
        if ( vid.valid() )
            return OneMeshIntersection{vid,mesh.points[vid]};
        if ( topology.dest( topology.prev( edgeOp.e ) ) == pVId )
            return OneMeshIntersection{edgeOp.e,mesh.edgePoint( edgeOp )};
        else
            return OneMeshIntersection{edgeOp.e.sym(),mesh.edgePoint( edgeOp )};
    }
    else if ( std::holds_alternative<EdgeId>( prev.primitiveId ) )
    {
        auto pEId = std::get<EdgeId>( prev.primitiveId );
        if ( std::holds_alternative<VertId>( next.primitiveId ) )
        {
            auto nVId = std::get<VertId>( next.primitiveId );
            if ( topology.dest( topology.next( pEId ) ) == nVId )
                return {};
            if ( topology.dest( pEId ) == nVId || topology.org( pEId ) == nVId || topology.dest( topology.prev( pEId ) ) == nVId )
            {
                assert( fromSameTriangle( topology, mesh.toTriPoint( nVId ), MeshTriPoint( MeshEdgePoint( pEId, 0.5f ) ) ) );
                return OneMeshIntersection{ findSharedFace( topology,nVId,pEId,curr ),mesh.triPoint( curr ) };
            }
        }
        else if ( std::holds_alternative<EdgeId>( next.primitiveId ) )
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
        auto vid = curr.inVertex( topology );
        if ( vid.valid() )
            return OneMeshIntersection{vid,mesh.points[vid]};
        if ( topology.prev( edgeOp.e ) == pEId || topology.next( edgeOp.e.sym() ) == pEId.sym() )
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
            res.primitiveId = v;
        else
            res.primitiveId = e.e;
    }
    else
        res.primitiveId = mesh.topology.left( mtp.e );
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
    MR_WRITER( mesh );
    HashMap<FaceId, std::vector<int>> face2contoursMap;
    for ( int i = 0; i < contours.size(); ++i )
    {
        FaceId f = std::get<FaceId>( contours[i].intersections.front().primitiveId );
        face2contoursMap[f].push_back( i );
    }
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
        mesh.splitFace( faceId, massCenter, nullptr, new2oldMap );
    }
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
    if ( outA )
        outA->resize( contours.size() );
    if ( outB )
        outB->resize( contours.size() );
    if ( outPtsA )
        outPtsA->resize( contours.size() );
    ParallelFor( contours, [&]( size_t j )
    {
        OneMeshContour curA, curB;
        Contour3f ptsA;
        const auto& curInContour = contours[j];
        if ( curInContour.empty() )
            return;
        curA.closed = curB.closed = isClosed( curInContour );
        if ( outA )
            curA.intersections.resize( curInContour.size() );
        if ( outB )
            curB.intersections.resize( curInContour.size() );
        if ( outPtsA )
            resizeNoInit( ptsA, curInContour.size() );

        ParallelFor( curInContour, [&]( size_t i )
        {
            Vector3f a, b, c, d, e;
            const auto& inIntersection = curInContour[i];
            OneMeshIntersection pntA, pntB;

            if ( inIntersection.isEdgeATriB() )
            {
                pntA.primitiveId = inIntersection.edge;
                pntB.primitiveId = inIntersection.tri();
                meshB.getTriPoints( inIntersection.tri(), a, b, c );
                d = meshA.orgPnt( inIntersection.edge );
                e = meshA.destPnt( inIntersection.edge );
            }
            else
            {
                pntB.primitiveId = inIntersection.edge;
                pntA.primitiveId = inIntersection.tri();
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

            if ( !curA.intersections.empty() )
                curA.intersections[i] = pntA;
            if ( !ptsA.empty() )
                ptsA[i] = pntA.coordinate;
            if ( !curB.intersections.empty() )
            {
                pntB.coordinate = rigidB2A ? inverseXf( pntA.coordinate ) : pntA.coordinate;
                curB.intersections[i] = pntB;
            }
        } );
        if ( !curA.closed && addSelfyTerminalVerts )
        {
            const auto & points = meshA.points;
            const auto & topology = meshA.topology;
            const auto i0 = curInContour.front();
            if ( topology.right( i0.edge ) )
            {
                const auto v0 = topology.dest( topology.prev( i0.edge ) );
                assert( topology.isTriVert( i0.tri(), v0 ) );
                OneMeshIntersection o0{ .primitiveId = v0, .coordinate = points[v0] };
                if ( !curA.intersections.empty() )
                    curA.intersections.insert( curA.intersections.begin(), o0 );
                if ( !ptsA.empty() )
                    ptsA.insert( ptsA.begin(), o0.coordinate );
                if ( !curB.intersections.empty() )
                    curB.intersections.insert( curB.intersections.begin(), o0 );
            }

            const auto i1 = curInContour.back();
            if ( topology.left( i1.edge ) )
            {
                const auto v1 = topology.dest( topology.next( i1.edge ) );
                assert( topology.isTriVert( i1.tri(), v1 ) );
                OneMeshIntersection o1{ .primitiveId = v1, .coordinate = points[v1] };
                if ( !curA.intersections.empty() )
                    curA.intersections.push_back( o1 );
                if ( !ptsA.empty() )
                    ptsA.push_back( o1.coordinate );
                if ( !curB.intersections.empty() )
                    curB.intersections.push_back( o1 );
            }
        }

        if ( outA )
            (*outA)[j] = std::move( curA );
        if ( outB )
            (*outB)[j] = std::move( curB );
        if ( outPtsA )
            (*outPtsA)[j] = std::move( ptsA );
    } );
}

OneMeshContours getOneMeshSelfIntersectionContours( const Mesh& mesh, const ContinuousContours& contours, const CoordinateConverters& converters, const AffineXf3f* rigidB2A /*= nullptr */ )
{
    MR_TIMER;
    OneMeshContours res;
    AffineXf3f inverseXf;
    if ( rigidB2A )
        inverseXf = rigidB2A->inverse();
    res.resize( contours.size() );
    for ( int j = 0; j < contours.size(); ++j )
    {
        auto& curOutContour = res[j].intersections;
        const auto& curInContour = contours[j];
        res[j].closed = isClosed( curInContour );
        curOutContour.resize( curInContour.size() );

        tbb::parallel_for( tbb::blocked_range<size_t>( 0, curInContour.size() ),
            [&] ( const tbb::blocked_range<size_t>& range )
        {
            Vector3f a, b, c, d, e;
            for ( size_t i = range.begin(); i < range.end(); ++i )
            {
                const auto& inIntersection = curInContour[i];
                auto& outIntersection = curOutContour[i];
                if ( !rigidB2A == inIntersection.isEdgeATriB() )
                    outIntersection.primitiveId = inIntersection.edge;
                else
                    outIntersection.primitiveId = inIntersection.tri();
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
    }
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

Expected<OneMeshContour> convertMeshTriPointsToMeshContour( const Mesh& mesh, const std::vector<MeshTriPoint>& meshTriPointsOrg,
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

    std::vector<int> sameEdgeMTPs;
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
    std::vector<OneMeshContour> surfacePaths( sizeMTP );
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
        auto partContours = convertSurfacePathsToMeshContours( mesh, { std::move( sp.value() ) } );
        assert( partContours.size() == 1 );
        surfacePaths[i] = partContours[0];
        if ( surfacePaths[i].intersections.size() == 1 )
        {
            // if lone make sure that prev MeshTriPoint is on the right (if on the left - sym)
            if ( std::holds_alternative<EdgeId>( surfacePaths[i].intersections[0].primitiveId ) )
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
                    auto inVert = meshTriPoints[i].inVertex( mesh.topology );
                    if ( !inVert )
                    {
                        // if mtp is on edge - make sure it is prev(e) or next(e.sym)
                        if ( mesh.topology.next( edge ).undirected() == onEdge.e.undirected() ||
                            mesh.topology.prev( edge.sym() ).undirected() == onEdge.e.undirected() )
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

    sameMtpsNavigator = 0;
    pivotNavigator = 0;
    const float closeEdgeEps = std::numeric_limits<float>::epsilon() * box.diagonal();
    // add interjacent
    for ( int i = 0; i < meshTriPoints.size(); ++i )
    {
        int realPivotIndex = -1;
        if ( pivotIndices )
        {
            while ( sameMtpsNavigator < sameEdgeMTPs.size() && pivotNavigator == sameEdgeMTPs[sameMtpsNavigator] )
            {
                ++pivotNavigator;
                ++sameMtpsNavigator;
            }
            realPivotIndex = pivotNavigator;
            ++pivotNavigator;
        }

        std::vector<OneMeshIntersection>* prevInter = ( closed || i > 0 ) ? &surfacePaths[( i + int( meshTriPoints.size() ) - 1 ) % meshTriPoints.size()].intersections : nullptr;
        const std::vector<OneMeshIntersection>* nextInter = ( i < sizeMTP ) ? &surfacePaths[i].intersections : nullptr;
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
                    if ( prevInter )
                        prevInter->back() = *centerInterOp;
                }
                else
                {
                    res.intersections.back() = *centerInterOp;
                    mtpPushed = true;
                }
            }
            if ( pivotIndices && mtpPushed )
            {
                int currentIndex = int( res.intersections.size() ) - 1;
                if ( pivotNavigator > 0 && ( *pivotIndices )[realPivotIndex - 1] == currentIndex )
                    ( *pivotIndices )[realPivotIndex - 1] = -1;
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
    if ( closed && !res.intersections.empty() )
    {
        res.intersections.push_back( res.intersections.front() );
        res.closed = true;
        if ( pivotIndices )
            ( *pivotIndices ).back() = ( *pivotIndices ).front();
    }
    return res;
}

Expected<OneMeshContour> convertMeshTriPointsToMeshContour( const Mesh& mesh, const std::vector<MeshTriPoint>& meshTriPointsOrg,
    SearchPathSettings searchSettings, std::vector<int>* pivotIndices )
{
    MeshTriPointsConnector conFn = [&] ( const MeshTriPoint& start, const MeshTriPoint& end, int, int )->Expected<SurfacePath>
    {
        auto res = computeGeodesicPath( mesh, start, end, searchSettings.geodesicPathApprox, searchSettings.maxReduceIters );
        if ( !res.has_value() )
            return unexpected( toString( res.error() ) );
        return *res;
    };
    return convertMeshTriPointsToMeshContour( mesh, meshTriPointsOrg, conFn, pivotIndices );
}

Expected<OneMeshContour> convertMeshTriPointsToClosedContour( const Mesh& mesh, const std::vector<MeshTriPoint>& meshTriPointsOrg,
    SearchPathSettings searchSettings, std::vector<int>* pivotIndices )
{
    auto conts = meshTriPointsOrg;
    conts.push_back( meshTriPointsOrg.front() );
    return convertMeshTriPointsToMeshContour( mesh, conts, searchSettings, pivotIndices );
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
            updatedPath.front() = startMep;
        for ( int i = 0; i < surfacePath.size(); ++i )
            updatedPath[i + shift] = surfacePath[i];
        if ( endMep )
            updatedPath.back() = endMep;
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

} //namespace MR
