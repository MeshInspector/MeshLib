#include "MRMeshFixer.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRRingIterator.h"
#include "MRBitSetParallelFor.h"
#include "MRTriMath.h"
#include "MRParallelFor.h"

namespace MR
{

// given a vertex, returns two edges with the origin in this vertex consecutive in the vertex ring without left faces both;
// both edges may be the same if there is only one edge without left face;
// or both edges can be invalid if all vertex edges have left face
static EdgePair getTwoSeqNoLeftAtVertex( const MeshTopology & m, VertId a )
{
    EdgeId e0 = m.edgeWithOrg( a );
    if ( !e0.valid() )
        return {}; //invalid vertex

    // find first hole edge
    EdgeId eh = e0;
    for (;;)
    {
        if ( !m.left( eh ).valid() )
            break;
        eh = m.next( eh );
        if ( eh == e0 )
            return {}; // no single hole near a
    }

    // find second hole edge
    for ( EdgeId e = m.next( eh ); e != e0; e = m.next( e ) )
    {
        if ( !m.left( e ).valid() )
            return { eh, e }; // another hole near a
    }

    return { eh, eh };
}

int duplicateMultiHoleVertices( Mesh & mesh )
{
    int duplicates = 0;
    const auto lastVert = mesh.topology.lastValidVert();
    for ( VertId v{0}; v <= lastVert; ++v )
    {
        auto ee = getTwoSeqNoLeftAtVertex( mesh.topology, v );
        if ( ee.first == ee.second )
            continue;

        EdgeId e1 = ee.first;
        EdgeId e0 = e1;
        while ( mesh.topology.right( e0 ).valid() )
            e0 = mesh.topology.prev( e0 );

        // unsplice [e0, e1] and create new vertex for it
        mesh.topology.splice( mesh.topology.prev( e0 ), e1 );
        assert( !mesh.topology.org( e0 ).valid() );

        auto vDup = mesh.addPoint( mesh.points[v] );
        mesh.topology.setOrg( e0, vDup );

        ++duplicates;
        --v;
    }

    return duplicates;
}

Expected<std::vector<MultipleEdge>> findMultipleEdges( const MeshTopology& topology, ProgressCallback cb )
{
    MR_TIMER
    tbb::enumerable_thread_specific<std::vector<MultipleEdge>> threadData;
    const VertId lastValidVert = topology.lastValidVert();
    
    auto mainThreadId = std::this_thread::get_id();
    std::atomic<bool> keepGoing{ true };
    std::atomic<size_t> numDone{ 0 };
    tbb::parallel_for( tbb::blocked_range<size_t>( size_t{ 0 },  size_t( lastValidVert ) + 1 ), [&] ( const tbb::blocked_range<size_t>& range )
    {
        auto & tls = threadData.local();
        std::vector<VertId> neis;
        for ( VertId v = VertId( range.begin() ); v < VertId( range.end() ); ++v )
        {
            if ( cb && !keepGoing.load( std::memory_order_relaxed ) )
                break;

            if ( !topology.hasVert( v ) )
                continue;
            neis.clear();
            for ( auto e : orgRing( topology, v ) )
            {
                auto nv = topology.dest( e );
                if ( nv > v )
                    neis.push_back( nv );
            }
            std::sort( neis.begin(), neis.end() );
            auto it = neis.begin();
            for (;;)
            {
                it = std::adjacent_find( it, neis.end() );
                if ( it == neis.end() )
                    break;
                auto nv = *it;
                tls.emplace_back( v, nv );
                assert( nv == *( it + 1 ) );
                ++++it;
                while ( it != neis.end() && *it == nv )
                    ++it;
                if ( it == neis.end() )
                    break;
            }
        }

        if ( cb )
            numDone += range.size();

        if ( cb && std::this_thread::get_id() == mainThreadId )
        {
            if ( !cb( float( numDone ) / float( lastValidVert + 1 ) ) )
                keepGoing.store( false, std::memory_order_relaxed );
        }
    } );

    if ( !keepGoing.load( std::memory_order_relaxed ) || ( cb && !cb( 1.0f ) ) )
        return unexpectedOperationCanceled();

    std::vector<MultipleEdge> res;
    for ( const auto & ns : threadData )
        res.insert( res.end(), ns.begin(), ns.end() );
    // sort the result to make it independent of mesh distribution among threads
    std::sort( res.begin(), res.end() );

    return res;
}

VertBitSet findNRingVerts( const MeshTopology& topology, int n, const VertBitSet* region /*= nullptr */ )
{
    const auto& zone = topology.getVertIds( region );
    VertBitSet result( zone.size() );
    BitSetParallelFor( zone, [&] ( VertId v )
    {
        int counter = 0;
        for ( auto e : orgRing( topology, v ) )
        {
            if ( !topology.left( e ) )
                return;
            ++counter;
            if ( counter > n )
                return;
        }
        if ( counter < n )
            return;
        assert( counter == n );
        result.set( v );
    } );
    return result;
}

void fixMultipleEdges( Mesh & mesh, const std::vector<MultipleEdge> & multipleEdges )
{
    if ( multipleEdges.empty() )
        return;
    MR_TIMER
    MR_WRITER( mesh )

    for ( const auto & mE : multipleEdges )
    {
        int num = 0;
        for ( auto e : orgRing( mesh.topology, mE.first ) )
        {
            if ( mesh.topology.dest( e ) != mE.second )
                continue;
            if ( num++ == 0 )
                continue; // skip the first edge in the group
            mesh.splitEdge( e.sym() );
        }
        assert( num > 1 ); //it was really multiply connected pair of vertices
    }
}

void fixMultipleEdges( Mesh & mesh )
{
    fixMultipleEdges( mesh, findMultipleEdges( mesh.topology ).value() );
}

Expected<FaceBitSet> findDegenerateFaces( const MeshPart& mp, float criticalAspectRatio, ProgressCallback cb )
{
    MR_TIMER
    FaceBitSet res( mp.mesh.topology.faceSize() );
    auto completed = BitSetParallelFor( mp.mesh.topology.getFaceIds( mp.region ), [&] ( FaceId f )
    {
        if ( !mp.mesh.topology.hasFace( f ) )
            return;
        if ( mp.mesh.triangleAspectRatio( f ) >= criticalAspectRatio )
            res.set( f );
    }, cb );

    if ( !completed )
        return unexpectedOperationCanceled();

    return res;
}

Expected<UndirectedEdgeBitSet> findShortEdges( const MeshPart& mp, float criticalLength, ProgressCallback cb )
{
    MR_TIMER
    const auto criticalLengthSq = sqr( criticalLength );
    UndirectedEdgeBitSet res( mp.mesh.topology.undirectedEdgeSize() );
    auto completed = BitSetParallelForAll( res, [&] ( UndirectedEdgeId ue )
    {
        if ( !mp.mesh.topology.isInnerOrBdEdge( ue, mp.region ) )
            return;
        if ( mp.mesh.edgeLengthSq( ue ) <= criticalLengthSq )
            res.set( ue );
    }, cb );    

    if ( !completed )
        return unexpectedOperationCanceled();

    return res;
}

bool isEdgeBetweenDoubleTris( const MeshTopology& topology, EdgeId e )
{
    return topology.next( e.sym() ) == topology.prev( e.sym() ) &&
        topology.isLeftTri( e ) && topology.isLeftTri( e.sym() );
}

EdgeId eliminateDoubleTris( MeshTopology& topology, EdgeId e, FaceBitSet * region )
{
    const auto ex = topology.next( e.sym() );
    const EdgeId ep = topology.prev( e );
    const EdgeId en = topology.next( e );
    if ( ex != topology.prev( e.sym() ) || ep == en || !topology.isLeftTri( e ) || !topology.isLeftTri( e.sym() ) )
        return {};
    // left( e ) and right( e ) are double triangles
    if ( auto f = topology.left( e ) )
    {
        if ( region )
            region->reset( f );
        topology.setLeft( e, {} );
    }
    if ( auto f = topology.left( e.sym() ) )
    {
        if ( region )
            region->reset( f );
        topology.setLeft( e.sym(), {} );
    }
    topology.setOrg( e.sym(), {} );
    topology.splice( e.sym(), ex );
    topology.splice( ep, e );
    assert( topology.isLoneEdge( e ) );
    topology.splice( en.sym(), ex.sym() );
    assert( topology.isLoneEdge( ex ) );
    topology.splice( ep, en );
    topology.splice( topology.prev( en.sym() ), en.sym() );
    assert( topology.isLoneEdge( en ) );
    return ep;
}

void eliminateDoubleTrisAround( MeshTopology & topology, VertId v, FaceBitSet * region )
{
    EdgeId e = topology.edgeWithOrg( v );
    EdgeId e0 = e;
    for (;;)
    {
        if ( auto ep = eliminateDoubleTris( topology, e, region ) )
            e0 = e = ep;
        else
        {
            e = topology.next( e );
            if ( e == e0 )
                break; // full ring has been inspected
            continue;
        }
    } 
}

bool isDegree3Dest( const MeshTopology& topology, EdgeId e )
{
    const EdgeId ex = topology.next( e.sym() );
    const EdgeId ey = topology.prev( e.sym() );
    return topology.next( ex ) == ey &&
        topology.isLeftTri( e ) && topology.isLeftTri( e.sym() ) && topology.isLeftTri( ex );
}

EdgeId eliminateDegree3Dest( MeshTopology& topology, EdgeId e, FaceBitSet * region )
{
    const EdgeId ex = topology.next( e.sym() );
    const EdgeId ey = topology.prev( e.sym() );
    const EdgeId ep = topology.prev( e );
    const EdgeId en = topology.next( e );
    if ( ep == en || topology.next( ex ) != ey ||
        !topology.isLeftTri( e ) || !topology.isLeftTri( e.sym() ) || !topology.isLeftTri( ex ) )
        return {};
    topology.flipEdge( ex );
    auto res = eliminateDoubleTris( topology, e, region );
    assert( res == ex );
    return res;
}

int eliminateDegree3Vertices( MeshTopology& topology, VertBitSet & region, FaceBitSet * fs )
{
    MR_TIMER
    auto candidates = region;
    int res = 0;
    for (;;)
    {
        const int x = res;
        for ( auto v : candidates )
        {
            candidates.reset( v );
            const auto e0 = topology.edgeWithOrg( v );
            if ( !isDegree3Dest( topology, e0.sym() ) )
                continue;
            ++res;
            region.reset( v );
            for ( auto e : orgRing( topology, e0 ) )
                if ( auto vn = topology.dest( e ); region.test( vn ) )
                    candidates.autoResizeSet( vn );
            [[maybe_unused]] auto ep = eliminateDegree3Dest( topology, e0.sym(), fs );
            assert( ep );
        }
        if ( res == x )
            break;
    }
    return res;
}

EdgeId isVertexRepeatedOnHoleBd( const MeshTopology& topology, VertId v )
{
    for ( EdgeId e0 : orgRing( topology, v ) )
    {
        if ( topology.left( e0 ) )
            continue;
        // not very optional in case of many boundary edges, but it shall be rare
        for ( EdgeId e1 : orgRing0( topology, e0 ) )
        {
            if ( topology.left( e1 ) )
                continue;
            if ( topology.fromSameLeftRing( e0, e1 ) )
                return e0;
        }
    }
    return {};
}

VertBitSet findRepeatedVertsOnHoleBd( const MeshTopology& topology )
{
    MR_TIMER
    const auto holeRepresEdges = topology.findHoleRepresentiveEdges();

    VertBitSet res;
    if ( holeRepresEdges.empty() )
        return res;

    struct ThreadData
    {
        explicit ThreadData( size_t vertSize ) : repeatedVerts( vertSize ), currHole( vertSize ) {}

        VertBitSet repeatedVerts;
        VertBitSet currHole;
    };

    tbb::enumerable_thread_specific<ThreadData> tls( topology.vertSize() );
    ParallelFor( holeRepresEdges, tls, [&]( size_t i, ThreadData & threadData )
    {
        const auto e0 = holeRepresEdges[i];
        for ( auto e : leftRing( topology, e0 ) )
        {
            auto v = topology.org( e );
            if ( threadData.currHole.test_set( v ) )
                threadData.repeatedVerts.set( v );
        }
        for ( auto e : leftRing( topology, e0 ) )
        {
            auto v = topology.org( e );
            threadData.currHole.reset( v );
        }
    } );

    for ( const auto & threadData : tls )
        res |= threadData.repeatedVerts;
    return res;
}

/// adds in complicatingFaces the faces not from the wedge with largest angle of faces connected by edges incident to given vertex
static void findHoleComplicatingFaces( const Mesh & mesh, VertId v, std::vector<FaceId> & complicatingFaces )
{
    EdgeId bd;
    float bdAngle = -1;
    
    auto angle = [&]( EdgeId e )
    {
        assert( !mesh.topology.left( e ) );
        float res = 0;
        while ( mesh.topology.right( e ) )
        {
            auto d1 = mesh.edgeVector( e );
            auto d0 = mesh.edgeVector( e = mesh.topology.prev( e ) );
            res += MR::angle( d0, d1 );
        }
        return res;
    };

    auto report = [&]( EdgeId e )
    {
        assert( !mesh.topology.left( e ) );
        while ( auto r = mesh.topology.right( e ) )
        {
            complicatingFaces.push_back( r );
            e = mesh.topology.prev( e );
        }
    };

    for ( EdgeId e : orgRing( mesh.topology, v ) )
    {
        if ( mesh.topology.left( e ) )
            continue;
        auto eAngle = angle( e );
        if ( eAngle <= bdAngle )
            report( e );
        else
        {
            if ( bd )
                report( bd );
            bd = e;
            bdAngle = eAngle;
        }
    }
}

FaceBitSet findHoleComplicatingFaces( const Mesh & mesh )
{
    MR_TIMER

    tbb::enumerable_thread_specific<std::vector<FaceId>> threadData;
    BitSetParallelFor( findRepeatedVertsOnHoleBd( mesh.topology ), [&]( VertId v )
    {
        findHoleComplicatingFaces( mesh, v, threadData.local() );
    } );

    FaceId maxFace;
    for ( const auto & fs : threadData )
        for ( FaceId f : fs )
            maxFace = std::max( maxFace, f );

    FaceBitSet res;
    res.resize( maxFace + 1 );
    for ( const auto & fs : threadData )
        for ( FaceId f : fs )
            res.set( f );
    return res;
}

} //namespace MR
