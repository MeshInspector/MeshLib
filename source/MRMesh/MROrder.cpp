#include "MROrder.h"
#include "MRMeshTopology.h"
#include "MRRingIterator.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"
#include <algorithm>
#if !defined(__APPLE__) && !defined(__EMSCRIPTEN__)
#include <execution>
#endif

namespace MR
{

VertexOrdering getVertexOrdering( const Buffer<FaceId> & invFaceMap, const MeshTopology & topology )
{
    MR_TIMER
    assert( topology.lastValidFace() < invFaceMap.size() );
    VertexOrdering res( topology.vertSize() );

    Timer t( "fill" );
    const auto getNewFace = [&]( FaceId f )
    {
        if ( !f )
            return f;
        return invFaceMap[ f ];
    };

    tbb::parallel_for( tbb::blocked_range<VertId>( 0_v, VertId{ topology.vertSize() } ),
    [&]( const tbb::blocked_range<VertId>& range )
    {
        for ( VertId v = range.begin(); v < range.end(); ++v )
        {
            auto f = ~std::uint32_t(0);
            for ( EdgeId e : orgRing( topology, v ) )
                f = std::min( f, std::uint32_t( getNewFace( topology.left( e ) ) ) );
            res[v] = OrderedVertex{ v, f };
        }
    } );

    t.restart( "sort" );
#if !defined(__APPLE__) && !defined(__EMSCRIPTEN__)
    std::sort( std::execution::par, res.data(), res.data() + res.size() );
#else
    std::sort( res.data(), res.data() + res.size() );
#endif
    res.resize( topology.numValidVerts() );

    return res;
}

EdgeOrdering getEdgeOrdering( const Buffer<FaceId> & invFaceMap, const MeshTopology & topology )
{
    MR_TIMER
    assert( topology.lastValidFace() < invFaceMap.size() );
    EdgeOrdering res( topology.undirectedEdgeSize() );

    Timer t( "fill" );
    const auto getNewFace = [&]( FaceId f )
    {
        if ( !f )
            return f;
        return invFaceMap[ f ];
    };

    std::atomic<int> notLoneEdges{0};
    tbb::parallel_for( tbb::blocked_range<UndirectedEdgeId>( 0_ue, UndirectedEdgeId{ topology.undirectedEdgeSize() } ),
    [&]( const tbb::blocked_range<UndirectedEdgeId>& range )
    {
        int myNotLoneEdges = 0;
        for ( UndirectedEdgeId ue = range.begin(); ue < range.end(); ++ue )
        {
            auto f = std::min(
                std::uint32_t( getNewFace( topology.left( ue ) ) ),
                std::uint32_t( getNewFace( topology.right( ue ) ) ) );
            res[ue] = OrderedEdge{ ue, f };
            if ( int(f) >= 0 )
                ++myNotLoneEdges;
        }
        notLoneEdges.fetch_add( myNotLoneEdges, std::memory_order_relaxed );
    } );

    t.restart( "sort" );
#if !defined(__APPLE__) && !defined(__EMSCRIPTEN__)
    std::sort( std::execution::par, res.data(), res.data() + res.size() );
#else
    std::sort( res.data(), res.data() + res.size() );
#endif
    res.resize( notLoneEdges );

    return res;
}

} //namespace MR
