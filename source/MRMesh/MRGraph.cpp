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
