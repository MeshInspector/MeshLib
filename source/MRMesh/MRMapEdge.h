#pragma once

#include "MRphmap.h"

namespace MR
{

inline EdgeId mapEdge( const WholeEdgeMap & map, EdgeId src )
{
    EdgeId res = map[ src.undirected() ];
    if ( res && src.odd() )
        res = res.sym();
    return res;
}

inline EdgeId mapEdge( const WholeEdgeHashMap & map, EdgeId src )
{
    EdgeId res;
    auto it = map.find( src.undirected() );
    if ( it != map.end() )
    {
        res = it->second;
        if ( src.odd() )
            res = res.sym();
    }
    return res;
}

} // namespace MR
