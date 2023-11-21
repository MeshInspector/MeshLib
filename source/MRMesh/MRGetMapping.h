#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// given input bit-set (src), converts each id corresponding to set bit using given map, and sets its bit in the resulting bit set
[[nodiscard]] MRMESH_API UndirectedEdgeBitSet getMapping( const UndirectedEdgeBitSet & src, const WholeEdgeMap & map );

/// given input bit-set (src), converts each id corresponding to set bit using given map, and sets its bit in the resulting bit set
[[nodiscard]] MRMESH_API UndirectedEdgeBitSet getMapping( const UndirectedEdgeBitSet & src, const WholeEdgeHashMap & map );

/// given input bit-set (src), converts each id corresponding to set bit using given map, and sets its bit in the resulting bit set
[[nodiscard]] MRMESH_API UndirectedEdgeBitSet getMapping( const UndirectedEdgeBitSet & src, const UndirectedEdgeBMap & map );

} //namespace MR
