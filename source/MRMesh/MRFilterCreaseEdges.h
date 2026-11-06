#pragma once
#include "MRMeshFwd.h"

namespace MR
{

 /// filters given edges using the following criteria:
 /// if \p filterComponents is true then connected components with summary length of their edges less than \p critLength will be excluded
 /// if \p filterBranches is true then branches shorter than \p critLength will be excluded
MRMESH_API void filterCreaseEdges( const Mesh& mesh, UndirectedEdgeBitSet& creaseEdges, float critLength, bool filterComponents = true, bool filterBranches = false );

} // namespace MR
