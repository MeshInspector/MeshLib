#pragma once

#include "MRMeshFwd.h"
#include <vector>

namespace MR
{

/// given triangle 0bc in 3D and line segment 0d in 2D, and |0b|=|0d|;
/// finds e, such that triangle 0bc is equal to 0de;
/// returns (0,0) if |0b|=0
template<typename T>
Vector2<T> unfoldOnPlane( const Vector3<T>& b, const Vector3<T>& c, const Vector2<T>& d, bool toLeftFrom0d );

/// \defgroup PlanarPathGroup Planar Path
/// \ingroup SurfacePathGroup
/// \{

/// given path s-v-e, tries to decrease its length by moving away from v
/// \param outPath intermediate locations between s and e will be added here
/// \param tmp elements will be temporary allocated here
/// \param cachePath as far as we need two sides unfold, cache one to reduce allocations
MRMESH_API bool reducePathViaVertex( const Mesh & mesh, const MeshTriPoint & start, VertId v, const MeshTriPoint & end, 
    std::vector<MeshEdgePoint> & outPath, std::vector<Vector2f> & tmp, std::vector<MeshEdgePoint>& cachePath );

/// converts any input surface path into geodesic path (so reduces its length): start-path-end;
/// returns actual number of iterations performed
MRMESH_API int reducePath( const Mesh & mesh, const MeshTriPoint & start, std::vector<MeshEdgePoint> & path, const MeshTriPoint & end, int maxIter = 5 );

/// \}

} // namespace MR
