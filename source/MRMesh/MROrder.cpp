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

VertBMap getVertexOrdering( const FaceBMap & faceMap, const MeshTopology & topology )
{
    MR_TIMER

    struct OrderedVertex
    {
        VertId v;
        std::uint32_t f; // the smallest nearby face
        bool operator <( const OrderedVertex & b ) const
            { return std::tie( f, v ) < std::tie( b.f, b.v ); } // order vertices by f
    };
    static_assert( sizeof( OrderedVertex ) == 8 );
    /// mapping: new vertex id -> old vertex id in v-field
    using VertexOrdering = Buffer<OrderedVertex, VertId>;

    assert( topology.lastValidFace() < faceMap.b.size() );
    VertexOrdering ord( topology.vertSize() );

    Timer t( "fill" );
    tbb::parallel_for( tbb::blocked_range<VertId>( 0_v, VertId{ topology.vertSize() } ),
    [&]( const tbb::blocked_range<VertId>& range )
    {
        for ( VertId v = range.begin(); v < range.end(); ++v )
        {
            auto f = ~std::uint32_t(0);
            for ( EdgeId e : orgRing( topology, v ) )
                f = std::min( f, std::uint32_t( getAt( faceMap.b, topology.left( e ) ) ) );
            ord[v] = OrderedVertex{ v, f };
        }
    } );

    t.restart( "sort" );
#if !defined(__APPLE__) && !defined(__EMSCRIPTEN__)
    std::sort( std::execution::par, ord.data(), ord.data() + ord.size() );
#else
    std::sort( ord.data(), ord.data() + ord.size() );
#endif

    VertBMap res;
    res.b.resize( topology.vertSize() );
    res.tsize = topology.numValidVerts();
    tbb::parallel_for( tbb::blocked_range<VertId>( 0_v, VertId{ topology.vertSize() } ),
    [&]( const tbb::blocked_range<VertId>& range )
    {
        for ( VertId v = range.begin(); v < range.end(); ++v )
        {
            res.b[ord[v].v] = v < res.tsize ? v : VertId{};
        }
    } );
    
    return res;
}

UndirectedEdgeBMap getEdgeOrdering( const FaceBMap & faceMap, const MeshTopology & topology )
{
    MR_TIMER

    struct OrderedEdge
    {
        UndirectedEdgeId ue;
        std::uint32_t f; // the smallest nearby face
        bool operator <( const OrderedEdge & b ) const
            { return std::tie( f, ue ) < std::tie( b.f, b.ue ); } // order vertices by f
    };
    static_assert( sizeof( OrderedEdge ) == 8 );
    /// mapping: new vertex id -> old vertex id in v-field
    using EdgeOrdering = Buffer<OrderedEdge, UndirectedEdgeId>;

    assert( topology.lastValidFace() < faceMap.b.size() );
    EdgeOrdering ord( topology.undirectedEdgeSize() );

    Timer t( "fill" );
    std::atomic<int> notLoneEdges{0};
    tbb::parallel_for( tbb::blocked_range<UndirectedEdgeId>( 0_ue, UndirectedEdgeId{ topology.undirectedEdgeSize() } ),
    [&]( const tbb::blocked_range<UndirectedEdgeId>& range )
    {
        int myNotLoneEdges = 0;
        for ( UndirectedEdgeId ue = range.begin(); ue < range.end(); ++ue )
        {
            auto f = std::min(
                std::uint32_t( getAt( faceMap.b, topology.left( ue ) ) ),
                std::uint32_t( getAt( faceMap.b, topology.right( ue ) ) ) );
            ord[ue] = OrderedEdge{ ue, f };
            if ( int(f) >= 0 )
                ++myNotLoneEdges;
        }
        notLoneEdges.fetch_add( myNotLoneEdges, std::memory_order_relaxed );
    } );

    t.restart( "sort" );
#if !defined(__APPLE__) && !defined(__EMSCRIPTEN__)
    std::sort( std::execution::par, ord.data(), ord.data() + ord.size() );
#else
    std::sort( ord.data(), ord.data() + ord.size() );
#endif

    UndirectedEdgeBMap res;
    res.b.resize( topology.undirectedEdgeSize() );
    res.tsize = notLoneEdges;
    tbb::parallel_for( tbb::blocked_range<UndirectedEdgeId>( 0_ue, UndirectedEdgeId{ topology.undirectedEdgeSize() } ),
    [&]( const tbb::blocked_range<UndirectedEdgeId>& range )
    {
        for ( UndirectedEdgeId ue = range.begin(); ue < range.end(); ++ue )
        {
            res.b[ord[ue].ue] = ue < res.tsize ? ue : UndirectedEdgeId{};
        }
    } );

    return res;
}

} //namespace MR
