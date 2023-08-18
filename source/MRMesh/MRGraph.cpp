#include "MRGraph.h"
#include "MRTimer.h"

namespace MR
{

void Graph::construct( NeighboursPerVertex neighboursPerVertex, EndsPerEdge endsPerEdge )
{
    MR_TIMER

    validVerts_.clear();
    validVerts_.resize( neighboursPerVertex.size(), true );
    neighboursPerVertex_ = std::move( neighboursPerVertex );

    validEdges_.clear();
    validEdges_.resize( endsPerEdge.size(), true );
    endsPerEdge_ = std::move( endsPerEdge );

    assert( checkValidity() );
}

Graph::EdgeId Graph::findEdge( VertId a, VertId b ) const
{
    assert( a.valid() && validVerts_.test( a ) );
    assert( b.valid() && validVerts_.test( b ) );
    assert( a != b );
    for ( EdgeId e : neighboursPerVertex_[a] )
    {
        if ( endsPerEdge_[e].otherEnd( a ) == b )
            return e;
    }
    return {};
}

void Graph::merge( VertId remnant, VertId dead, std::function<void(EdgeId, EdgeId)> onMergeEdges )
{
    assert( remnant.valid() && validVerts_.test( remnant ) );
    assert( dead.valid() && validVerts_.test( dead ) );
    assert( remnant != dead );
    validVerts_.reset( dead );

    struct NeiEdge
    {
        VertId nei;
        EdgeId e;
        auto operator<=>(const NeiEdge&) const = default;
    };
    std::vector<NeiEdge> neiEdges;
    neiEdges.reserve( neighboursPerVertex_[remnant].size() + neighboursPerVertex_[dead].size() );
    for ( auto e : neighboursPerVertex_[remnant] )
    {
        const auto ends = endsPerEdge_[e];
        const auto nei = ends.otherEnd( remnant );
        if ( nei == dead )
        {
            validEdges_.reset( e );
            continue;
        }
        neiEdges.push_back( { nei, e } );
    }
    for ( auto e : neighboursPerVertex_[dead] )
    {
        auto & ends = endsPerEdge_[e];
        const auto nei = ends.otherEnd( dead );
        if ( nei == remnant )
        {
            validEdges_.reset( e );
            continue;
        }
        ends.replaceEnd( dead, remnant );
        neiEdges.push_back( { nei, e } );
    }
    std::sort( neiEdges.begin(), neiEdges.end() );

    // reuse the memory for neighbors
    Neighbours neis;
    if ( neighboursPerVertex_[remnant].size() >= neighboursPerVertex_[dead].size() )
        neis = std::move( neighboursPerVertex_[remnant] );
    else
        neis = std::move( neighboursPerVertex_[dead] );
    neis.clear();
    neighboursPerVertex_[dead] = {};

    for ( const auto & x : neiEdges )
    {
        if ( neis.empty() || endsPerEdge_[neis.back()].otherEnd( remnant ) != x.nei )
        {
            assert( validEdges_.test( x.e ) );
            neis.push_back( x.e );
            continue;
        }
        validEdges_.reset( x.e );
        [[maybe_unused]] auto n = erase( neighboursPerVertex_[x.nei], x.e );
        assert( n == 1 );
        onMergeEdges( neis.back(), x.e );
    }
    std::sort( neis.begin(), neis.end() );
    neighboursPerVertex_[remnant] = std::move( neis );
    //assert( checkValidity() );
}

bool Graph::checkValidity() const
{
    MR_TIMER

    #define CHECK(x) { assert(x); if (!(x)) return false; }

    CHECK( validVerts_.size() == neighboursPerVertex_.size() );
    CHECK( validEdges_.size() == endsPerEdge_.size() );

    for ( auto v : validVerts_ )
    {
        const auto & neis = neighboursPerVertex_[v];
        CHECK( std::is_sorted( neis.begin(), neis.end() ) );
        for ( auto e : neis )
        {
            CHECK( e.valid() );
            CHECK( e < endsPerEdge_.size() );
            CHECK( validEdges_.test( e ) );
            const auto ends = endsPerEdge_[e];
            CHECK( ends.v0 == v || ends.v1 == v );
            const auto w = ends.otherEnd( v );
            CHECK( e == findEdge( v, w ) );
            CHECK( e == findEdge( w, v ) );
        }
    }

    for ( auto e : validEdges_ )
    {
        const auto ends = endsPerEdge_[e];
        CHECK( ends.v0 && ends.v1 );
        CHECK( ends.v0 < ends.v1 );
        CHECK( ends.v0 < neighboursPerVertex_.size() );
        CHECK( validVerts_.test( ends.v0 ) );
        CHECK( ends.v1 < neighboursPerVertex_.size() );
        CHECK( validVerts_.test( ends.v1 ) );

        CHECK( e == findEdge( ends.v0, ends.v1 ) );
        CHECK( e == findEdge( ends.v1, ends.v0 ) );
    }

    return true;
}

} //namespace MR
