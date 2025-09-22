#pragma once

#include "MRMapOrHashMap.h"
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
[[nodiscard]] inline UndirectedEdgeId mapEdge( const WholeEdgeMap & map, UndirectedEdgeId src )
{
    EdgeId eres = map[ src ];
    return eres ? eres.undirected() : UndirectedEdgeId{};
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
[[nodiscard]] inline UndirectedEdgeId mapEdge( const WholeEdgeHashMap & map, UndirectedEdgeId src )
{
    auto it = map.find( src );
    return it != map.end() ? it->second.undirected() : UndirectedEdgeId{};
}

/// given input edge (src), converts its id using given map
[[nodiscard]] inline EdgeId mapEdge( const WholeEdgeMapOrHashMap & m, EdgeId src )
{
    return std::visit( overloaded{
        [src]( const WholeEdgeMap& map ) { return mapEdge( map, src ); },
        [src]( const WholeEdgeHashMap& hashMap ) { return mapEdge( hashMap, src ); }
    }, m.var );
}

/// given input edge (src), converts its id using given map
[[nodiscard]] inline UndirectedEdgeId mapEdge( const WholeEdgeMapOrHashMap & m, UndirectedEdgeId src )
{
    return std::visit( overloaded{
        [src]( const WholeEdgeMap& map ) { return mapEdge( map, src ); },
        [src]( const WholeEdgeHashMap& hashMap ) { return mapEdge( hashMap, src ); }
    }, m.var );
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
