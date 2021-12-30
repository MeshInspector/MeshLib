#pragma once

#include "MRMeshFwd.h"
#include <vector>

namespace MR
{

// given path s-v-e, tries to decrease its length by moving away from v
MRMESH_API bool reducePathViaVertex( const Mesh & mesh, const MeshTriPoint & start, VertId v, const MeshTriPoint & end, 
    std::vector<MeshEdgePoint> & outPath, // intermediate locations between s and e will be added here
    std::vector<Vector2f> & tmp ); // elements will be temporary allocated here

// reduces the length of given surface path: s-path-e
MRMESH_API void reducePath( const Mesh & mesh, const MeshTriPoint & start, std::vector<MeshEdgePoint> & path, const MeshTriPoint & end, int maxIter = 5 );

} //namespace MR
