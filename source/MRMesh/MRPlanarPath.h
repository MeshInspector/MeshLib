#pragma once

#include "MRMeshFwd.h"
#include <vector>

namespace MR
{

/// \defgroup PlanarPathGroup Planar Path
/// \ingroup SurfacePathGroup
/// \{

/// given path s-v-e, tries to decrease its length by moving away from v
/// \param outPath intermediate locations between s and e will be added here
/// \param tmp elements will be temporary allocated here
/// \param cachePath as far as we need to sides unfold, cache one to reduce allocations
MRMESH_API bool reducePathViaVertex( const Mesh & mesh, const MeshTriPoint & start, VertId v, const MeshTriPoint & end, 
    std::vector<MeshEdgePoint> & outPath, std::vector<Vector2f> & tmp, std::vector<MeshEdgePoint>& cachePath );

/// reduces the length of given surface path: start-path-end
MRMESH_API void reducePath( const Mesh & mesh, const MeshTriPoint & start, std::vector<MeshEdgePoint> & path, const MeshTriPoint & end, int maxIter = 5 );

/// \}

} // namespace MR
