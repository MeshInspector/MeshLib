#include "MRInTreePathBuilder.h"
#include "MRMeshTopology.h"
#include "MRRingIterator.h"
#include "MREdgePaths.h"
#include "MRTimer.h"

namespace MR
{

InTreePathBuilder::InTreePathBuilder( const MeshTopology & topology, const UndirectedEdgeBitSet & treeEdges )
    : topology_( topology )
    , treeEdges_( treeEdges )
{
    MR_TIMER
    vertDistance_.resize( topology_.vertSize(), -1 );

    VertBitSet unvisited = topology_.getValidVerts();
    std::vector<VertId> active;
    for ( auto vRoot : unvisited )
    {
        unvisited.reset( vRoot );
        vertDistance_[vRoot] = 0;
        active.push_back( vRoot );
        while( !active.empty() )
        {
            const auto v0 = active.back();
            active.pop_back();
            const auto d0 = vertDistance_[v0];
            for ( auto e : orgRing( topology_, v0 ) )
            {
                if ( !treeEdges_.test( e ) )
                    continue;
                const auto v = topology_.dest( e );
                if ( !unvisited.test_set( v, false ) )
                    continue;
                auto & d = vertDistance_[v];
                assert( d < 0 );
                d = d0 + 1;
                active.push_back( v );
            }
        }
    }
}

EdgeId InTreePathBuilder::getEdgeBack_( VertId v0 ) const
{
    const auto d0 = vertDistance_[v0];
    assert( d0 >= 0 );
    for ( auto e : orgRing( topology_, v0 ) )
    {
        if ( !treeEdges_.test( e ) )
            continue;
        const auto v = topology_.dest( e );
        const auto d = vertDistance_[v];
        assert( d >= 0 );
        if ( d0 == d + 1 )
            return e;
    }
    return {};
}

EdgePath InTreePathBuilder::build( VertId start, VertId finish ) const
{
    MR_TIMER
    EdgePath res;
    auto ds = vertDistance_[start];
    if ( ds < 0 )
    {
        assert( false );
        return res;
    }
    auto df = vertDistance_[finish];
    if ( df < 0 )
    {
        assert( false );
        return res;
    }

    EdgePath start2branch;
    start2branch.reserve( ds );
    
    auto oneStartStep = [&]()
    {
        assert( ds > 0 );
        const auto e = getEdgeBack_( start );
        assert( e );
        start2branch.push_back( e );
        --ds;
        assert( start == topology_.org( e ) );
        start = topology_.dest( e );
        assert( ds == vertDistance_[start] );
    };

    EdgePath finish2branch;
    finish2branch.reserve( df );

    auto oneFinishStep = [&]()
    {
        assert( df > 0 );
        const auto e = getEdgeBack_( finish );
        assert( e );
        finish2branch.push_back( e );
        --df;
        assert( finish == topology_.org( e ) );
        finish = topology_.dest( e );
        assert( df == vertDistance_[finish] );
    };

    while ( ds > df )
        oneStartStep();

    while ( df > ds )
        oneFinishStep();

    assert( ds == df );
    while ( start != finish )
    {
        if ( ds == 0 )
        {
            assert( false ); // no path exists
            return res;
        }
        oneStartStep();
        oneFinishStep();
        assert( ds == df );
    }

    res = std::move( start2branch );
    res.reserve( res.size() + finish2branch.size() );
    for ( int i = (int)finish2branch.size() - 1; i >= 0; --i )
        res.push_back( finish2branch[i].sym() );
    assert( isEdgePath( topology_, res ) );

    return res;
}

} //namespace MR
