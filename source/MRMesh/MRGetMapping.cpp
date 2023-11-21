#include "MRGetMapping.h"
#include "MRBitSet.h"
#include "MRBuffer.h"
#include "MRVector.h"

namespace MR
{

UndirectedEdgeBitSet getMapping( const UndirectedEdgeBitSet & src, const WholeEdgeMap & map )
{
    UndirectedEdgeBitSet res;
    for ( auto b : src )
        if ( auto mapped = map[b] )
            res.autoResizeSet( mapped.undirected() );
    return res;
}

UndirectedEdgeBitSet getMapping( const UndirectedEdgeBitSet & src, const WholeEdgeHashMap & map )
{
    UndirectedEdgeBitSet res;
    for ( auto b : src )
        if ( auto mapped = getAt( map, b ) )
            res.autoResizeSet( mapped.undirected() );
    return res;
}

UndirectedEdgeBitSet getMapping( const UndirectedEdgeBitSet & src, const UndirectedEdgeBMap & map )
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
