#include "MRWatershedGraph.h"
#include "MRMeshTopology.h"
#include "MRphmap.h"
#include "MRRingIterator.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
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

    outsideId_ = Graph::VertId( numBasins );
    ++numBasins;
    basins_.resize( numBasins );
    Graph::NeighboursPerVertex neighboursPerVertex( numBasins );
    parentBasin_.clear();
    parentBasin_.reserve( numBasins );
    for ( Graph::VertId v( 0 ); v < numBasins; ++v )
        parentBasin_.push_back( v );
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
            Graph::VertId basin( l ? face2basin[l] : outsideId_ );
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
            const Graph::VertId basinL( l ? face2basin[l] : outsideId_ );
            auto & infoL = basins_[basinL];
            if ( h < getHeightAt( infoL.lowestVert ) )
                infoL.lowestVert = v;
            auto r = topology.right( e );
            const Graph::VertId basinR( r ? face2basin[r] : outsideId_ );
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

Graph::VertId WatershedGraph::getRootBasin( Graph::VertId v ) const
{
    assert( v );
    for (;;)
    {
        auto p = parentBasin_[v];
        if ( p == v )
            return v;
        v = p;
    }
}

void WatershedGraph::setParentsToRoots()
{
    MR_TIMER
    ParallelFor( parentBasin_, [&]( Graph::VertId v )
    {
        parentBasin_[v] = getRootBasin( v );
    } );
}

MRMESH_API std::pair<Graph::EdgeId, float> WatershedGraph::findLowestBd() const
{
    MR_TIMER
    Graph::EdgeId lowestEdge;
    float lowestLevel = FLT_MAX;
    for ( auto ei : graph_.validEdges() )
    {
        const auto ends = graph_.ends( ei );
        if ( ends.v0 == outsideId_ || ends.v1 == outsideId_ )
            continue;
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

Graph::VertId WatershedGraph::merge( Graph::VertId v0, Graph::VertId v1 )
{
    MR_TIMER
    assert( v0 && v1 );
    assert( v0 != outsideId_ && v1 != outsideId_ );
    assert( graph_.valid( v0 ) && graph_.valid( v1 ) );
    if ( v0 == v1 )
        return v0;

    assert( parentBasin_[v1] == v1 );
    parentBasin_[v1] = v0;
    if ( getHeightAt( basins_[v1].lowestVert ) < getHeightAt( basins_[v0].lowestVert ) )
        basins_[v0].lowestVert = basins_[v1].lowestVert;

    graph_.merge( v0, v1, [&]( Graph::EdgeId eremnant, Graph::EdgeId edead )
    {
        if ( getHeightAt( bds_[edead].lowestVert ) < getHeightAt( bds_[eremnant].lowestVert ) )
            bds_[eremnant].lowestVert = bds_[edead].lowestVert;
    } );
    return v0;
}

Graph::VertId WatershedGraph::mergeViaBd( Graph::EdgeId bd )
{
    assert( bd );
    const auto ends = graph_.ends( bd );
    return merge( ends.v0, ends.v1 );
}

FaceBitSet WatershedGraph::getBasinFaces( Graph::VertId basin ) const
{
    MR_TIMER
    FaceBitSet res;
    if ( basin == outsideId_ )
        return res;
    res.resize( topology_.faceSize() );
    assert( graph_.valid( basin ) );
    assert( basin == parentBasin_[basin] );
    BitSetParallelForAll( res, [&]( FaceId f )
    {
        if ( basin == getRootBasin( Graph::VertId( face2iniBasin_[f] ) ) )
            res.set( f );
    } );
    return res;
}

FaceBitSet WatershedGraph::getBasinFacesBelowLevel( Graph::VertId basin, float waterLevel ) const
{
    MR_TIMER
    FaceBitSet res;
    if ( basin == outsideId_ )
        return res;
    res.resize( topology_.faceSize() );
    assert( graph_.valid( basin ) );
    assert( basin == parentBasin_[basin] );
    BitSetParallelForAll( res, [&]( FaceId f )
    {
        if ( basin != getRootBasin( Graph::VertId( face2iniBasin_[f] ) ) )
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
    BitSetParallelForAll( res, [&]( UndirectedEdgeId ue )
    {
        auto l = topology_.left( ue );
        if ( !l )
            return;
        auto r = topology_.right( ue );
        if ( !r )
            return;
        const auto lBasin = getRootBasin( Graph::VertId( face2iniBasin_[l] ) );
        const auto rBasin = getRootBasin( Graph::VertId( face2iniBasin_[r] ) );
        if ( lBasin != rBasin )
            res.set( ue );
    } );
    return res;
}

} //namespace MR
