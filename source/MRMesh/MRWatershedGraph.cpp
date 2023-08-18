#include "MRWatershedGraph.h"
#include "MRMeshTopology.h"
#include "MRphmap.h"
#include "MRRingIterator.h"
#include "MRTimer.h"

namespace std
{

template<> 
struct hash<MR::Graph::EndVertices> 
{
    size_t operator()( MR::Graph::EndVertices const& e ) const noexcept
    {
        std::uint32_t x;
        std::uint32_t y;
        static_assert( sizeof( e.v0 ) == sizeof( std::uint32_t ) && sizeof( e.v1 ) == sizeof( std::uint32_t ) );
        std::memcpy( &x, &e.v0, sizeof( std::uint32_t ) );
        std::memcpy( &y, &e.v1, sizeof( std::uint32_t ) );
        return size_t( x ) ^ ( size_t( y ) << 16 );
    }
};

} // namespace std

namespace MR
{

void WatershedGraph::construct( const MeshTopology & topology, const VertScalars & heights, const Vector<int, FaceId> & face2basin, int numBasins )
{
    MR_TIMER
    assert( numBasins >= 0 );
    basins_.clear();
    bds_.clear();

    basins_.resize( numBasins );
    Graph::NeighboursPerVertex neighboursPerVertex( numBasins );
    Graph::EndsPerEdge endsPerEdge;

    HashMap<Graph::EndVertices, Graph::EdgeId> neiBasins2edge;

    for ( auto v : topology.getValidVerts() )
    {
        const auto h = heights[v];
        bool bdVert = false;
        Graph::VertId basin0;
        for ( auto e : orgRing( topology, v ) )
        {
            auto l = topology.left( e );
            if ( !l )
                continue;
            Graph::VertId basin( face2basin[l] );
            if ( !basin0 )
            {
                basin0 = basin;
                continue;
            }
            if ( basin != basin0 )
            {
                bdVert = true;
                break;
            }
        }
        if ( !bdVert )
        {
            if ( basin0 )
            {
                auto & info0 = basins_[basin0];
                info0.lowestHeight = std::min( info0.lowestHeight, h );
            }
            continue;
        }
        for ( auto e : orgRing( topology, v ) )
        {
            auto l = topology.left( e );
            if ( !l )
                continue;
            const Graph::VertId basinL( face2basin[l] );
            auto & infoL = basins_[basinL];
            infoL.lowestHeight = std::min( infoL.lowestHeight, h );
            auto r = topology.right( e );
            if ( !r )
                continue;
            const Graph::VertId basinR( face2basin[r] );
            if ( basinL == basinR )
                continue;

            Graph::EndVertices ends{ basinL, basinR };
            if ( ends.v0 > ends.v1 )
                std::swap( ends.v0, ends.v1 );

            auto [it, inserted] = neiBasins2edge.insert( { ends, endsPerEdge.endId() } );
            auto bdEdge = it->second;
            if ( inserted )
            {
                endsPerEdge.push_back( ends );
                bds_.emplace_back();
                neighboursPerVertex[basinL].push_back( bdEdge );
                neighboursPerVertex[basinR].push_back( bdEdge );
            }
            auto & bd = bds_[bdEdge];
            bd.lowestHeight = std::min( bd.lowestHeight, h );
        }
    }

    graph_.construct( std::move( neighboursPerVertex ), std::move( endsPerEdge ) );
}

void WatershedGraph::mergeViaLowestBd()
{
    MR_TIMER

    Graph::EdgeId lowestEdge;
    float lowestLevel = FLT_MAX;
    for ( auto ei : graph_.validEdges() )
    {
        const auto ends = graph_.ends( ei );
        const auto l0 = basins_[ends.v0].lowestHeight;
        const auto l1 = basins_[ends.v1].lowestHeight;
        const auto le = bds_[ei].lowestHeight;
        assert( le >= l0 && le >= l1 );
        const auto level = std::min( le - l0, le - l1 );
        if ( level < lowestLevel )
        {
            lowestLevel = level;
            lowestEdge = ei;
        }
    }
    if ( !lowestEdge )
        return;

    const auto ends = graph_.ends( lowestEdge );
    const auto vremnant = ends.v0;
    const auto vdead = ends.v1;
    {
        auto & lremnant = basins_[vremnant].lowestHeight;
        lremnant = std::min( lremnant, basins_[vdead].lowestHeight );
    }
    graph_.merge( vremnant, vdead, [&]( Graph::EdgeId remnant, Graph::EdgeId dead )
    {
        auto & lremnant = bds_[remnant].lowestHeight;
        lremnant = std::min( lremnant, bds_[dead].lowestHeight );
    } );
}

} //namespace MR
