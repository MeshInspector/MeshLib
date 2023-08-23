#include "MRWatershedGraph.h"
#include "MRMeshTopology.h"
#include "MRphmap.h"
#include "MRRingIterator.h"
#include "MRBitSetParallelFor.h"
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

WatershedGraph::WatershedGraph( const MeshTopology & topology, const VertScalars & heights, const Vector<int, FaceId> & face2basin, int numBasins )
    : topology_( topology )
    , heights_( heights )
    , face2iniBasin_( face2basin )
{
    MR_TIMER
    assert( numBasins >= 0 );
    basins_.clear();
    bds_.clear();

    basins_.resize( numBasins );
    Graph::NeighboursPerVertex neighboursPerVertex( numBasins );
    ufBasins_.reset( numBasins );
    Graph::EndsPerEdge endsPerEdge;

    HashMap<Graph::EndVertices, Graph::EdgeId> neiBasins2edge;

    for ( auto v : topology.getValidVerts() )
    {
        const auto h = heights_[v];
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
                if ( h < getHeightAt( info0.lowestVert ) )
                    info0.lowestVert = v;
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
            if ( h < getHeightAt( infoL.lowestVert ) )
                infoL.lowestVert = v;
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
            if ( h < getHeightAt( bd.lowestVert ) )
                bd.lowestVert = v;
        }
    }

    graph_.construct( std::move( neighboursPerVertex ), std::move( endsPerEdge ) );
}

MRMESH_API std::pair<Graph::EdgeId, float> WatershedGraph::findLowestBd() const
{
    MR_TIMER
    Graph::EdgeId lowestEdge;
    float lowestLevel = FLT_MAX;
    for ( auto ei : graph_.validEdges() )
    {
        const auto ends = graph_.ends( ei );
        const auto l0 = getHeightAt( basins_[ends.v0].lowestVert );
        const auto l1 = getHeightAt( basins_[ends.v1].lowestVert );
        const auto le = getHeightAt( bds_[ei].lowestVert );
        assert( le >= l0 && le >= l1 );
        const auto level = std::min( le - l0, le - l1 );
        if ( level < lowestLevel )
        {
            lowestLevel = level;
            lowestEdge = ei;
        }
    }
    return { lowestEdge, lowestLevel };
}

void WatershedGraph::mergeViaBd( Graph::EdgeId bd )
{
    MR_TIMER
    if ( !bd )
        return;

    const auto ends = graph_.ends( bd );
    const auto [vremnant, united] = ufBasins_.unite( ends.v0, ends.v1 );
    assert( united );
    const auto vdead = ends.otherEnd( vremnant );
    {
        if ( getHeightAt( basins_[vdead].lowestVert ) < getHeightAt( basins_[vremnant].lowestVert ) )
            basins_[vremnant].lowestVert = basins_[vdead].lowestVert;
    }
    graph_.merge( vremnant, vdead, [&]( Graph::EdgeId eremnant, Graph::EdgeId edead )
    {
        if ( getHeightAt( bds_[edead].lowestVert ) < getHeightAt( bds_[eremnant].lowestVert ) )
            bds_[eremnant].lowestVert = bds_[edead].lowestVert;
    } );
}

FaceBitSet WatershedGraph::getBasinFaces( Graph::VertId basin ) const
{
    MR_TIMER
    FaceBitSet res( topology_.faceSize() );
    const auto & roots = ufBasins_.roots();
    assert( basin == roots[basin] );
    BitSetParallelForAll( res, [&]( FaceId f )
    {
        if ( basin == roots[Graph::VertId( face2iniBasin_[f] )] )
            res.set( f );
    } );
    return res;
}

FaceBitSet WatershedGraph::getBasinFacesBelowLevel( Graph::VertId basin, float waterLevel ) const
{
    MR_TIMER
    FaceBitSet res( topology_.faceSize() );
    const auto & roots = ufBasins_.roots();
    assert( basin == roots[basin] );
    BitSetParallelForAll( res, [&]( FaceId f )
    {
        if ( basin != roots[Graph::VertId( face2iniBasin_[f] )] )
            return;
        VertId vs[3];
        topology_.getTriVerts( f, vs );
        for ( int i = 0; i < 3; ++i )
            if ( heights_[vs[i]] < waterLevel )
            {
                res.set( f );
                break;
            }
    } );
    return res;
}

UndirectedEdgeBitSet WatershedGraph::getInterBasinEdges() const
{
    MR_TIMER
    UndirectedEdgeBitSet res( topology_.undirectedEdgeSize() );
    const auto & roots = ufBasins_.roots();
    BitSetParallelForAll( res, [&]( UndirectedEdgeId ue )
    {
        auto l = topology_.left( ue );
        if ( !l )
            return;
        auto r = topology_.right( ue );
        if ( !r )
            return;
        const auto lBasin = roots[Graph::VertId( face2iniBasin_[l] )];
        const auto rBasin = roots[Graph::VertId( face2iniBasin_[r] )];
        if ( lBasin != rBasin )
            res.set( ue );
    } );
    return res;
}

} //namespace MR
