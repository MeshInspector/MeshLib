#pragma once

#include "MRphmap.h"
#include "MRVector.h"
#include "MRBuffer.h"

namespace MR
{

/// given input edge (src), converts its id using given map
[[nodiscard]] inline EdgeId mapEdge( const WholeEdgeMap & map, EdgeId src )
{
    EdgeId res = map[ src.undirected() ];
    if ( res && src.odd() )
        res = res.sym();
    return res;
}

/// given input edge (src), converts its id using given map
[[nodiscard]] inline EdgeId mapEdge( const WholeEdgeHashMap & map, EdgeId src )
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

/// given input edge (src), converts its id using given map
[[nodiscard]] inline UndirectedEdgeId mapEdge( const UndirectedEdgeBMap & map, UndirectedEdgeId src )
{
    return getAt( map.b, src );
}

/// given input bit-set (src), converts each id corresponding to set bit using given map, and sets its bit in the resulting bit set
[[nodiscard]] MRMESH_API UndirectedEdgeBitSet mapEdges( const WholeEdgeMap & map, const UndirectedEdgeBitSet & src );

/// given input bit-set (src), converts each id corresponding to set bit using given map, and sets its bit in the resulting bit set
[[nodiscard]] MRMESH_API UndirectedEdgeBitSet mapEdges( const WholeEdgeHashMap & map, const UndirectedEdgeBitSet & src );

/// given input bit-set (src), converts each id corresponding to set bit using given map, and sets its bit in the resulting bit set
[[nodiscard]] MRMESH_API UndirectedEdgeBitSet mapEdges( const UndirectedEdgeBMap & map, const UndirectedEdgeBitSet & src );

} // namespace MR
