#pragma once
#include "MRMeshFwd.h"

namespace MR
{

 /// filters given edges using the following criteria:
 /// connected components with summary length of their edges less than \param critLength will be excluded
 /// if \param filterBranches is true then branches shorter than \param critLength will be excluded
MRMESH_API void filterCreaseEdges( const Mesh& mesh, UndirectedEdgeBitSet& creaseEdges, float critLength, bool filterBranches = false );

} // namespace MR
