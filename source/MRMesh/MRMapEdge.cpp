#include "MRMapEdge.h"
#include "MRBitSet.h"
#include "MRBuffer.h"

namespace MR
{

UndirectedEdgeBitSet mapEdges( const WholeEdgeMap & map, const UndirectedEdgeBitSet & src )
{
    UndirectedEdgeBitSet res;
    for ( auto b : src )
        if ( auto mapped = map[b] )
        {
            const auto ue = mapped.undirected();
            if ( ue >= res.size() )
                res.resize( ue + 1 );
            res.set( ue );
        }
    return res;
}

UndirectedEdgeBitSet mapEdges( const WholeEdgeHashMap & map, const UndirectedEdgeBitSet & src )
{
    UndirectedEdgeBitSet res;
    for ( auto b : src )
        if ( auto mapped = getAt( map, b ) )
        {
            const auto ue = mapped.undirected();
            if ( ue >= res.size() )
                res.resize( ue + 1 );
            res.set( ue );
        }
    return res;
}

UndirectedEdgeBitSet mapEdges( const UndirectedEdgeBMap & map, const UndirectedEdgeBitSet & src )
{
    UndirectedEdgeBitSet res;
    if ( !src.any() )
        return res;
    res.resize( map.tsize );
    for ( auto b : src )
        if ( auto mapped = getAt( map.b, b ) )
            res.set( mapped );
    return res;
}

} //namespace MR
