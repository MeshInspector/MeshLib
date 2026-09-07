#pragma once

#include "MRMeshFwd.h"
#include <vector>

namespace MR
{

/// \defgroup PlanarPathGroup Planar Path
/// \ingroup SurfacePathGroup
/// \{

/// given triangle 0bc in 3D and line segment 0d in 2D, and |0b|=|0d|;
/// finds e, such that triangle 0bc is equal to 0de;
/// returns (0,0) if |0b|=0
template<typename T>
Vector2<T> unfoldOnPlane( const Vector3<T>& b, const Vector3<T>& c, const Vector2<T>& d, bool toLeftFrom0d );

/// given two 3D triangles ABC and ACD with shared edge AC,
/// returns the relative position x in [0,1] on edge AC (x=0 means A and x=1 means C),
/// where the shortest path from B to D crosses edge AC
template<typename T>
T shortestPathInQuadrangle( const Vector3<T>& a, const Vector3<T>& b, const Vector3<T>& c, const Vector3<T>& d );

/// given two 3D triangles ABC and ACD with shared edge AC,
/// returns true if after unfolding into plane they form a convex quadrangle
template<typename T>
bool isUnfoldQuadrangleConvex( const Vector3<T>& a, const Vector3<T>& b, const Vector3<T>& c, const Vector3<T>& d )
{
    auto x = shortestPathInQuadrangle( a, b, c, d );
    return x > 0 && x < 1; // if path goes via vertices A or C then the quadrangle is concave or degenerated to triangle
}

/// given path s-v-e, tries to decrease its length by moving away from v
/// \param outPath intermediate locations between s and e will be added here
/// \param tmp elements will be temporary allocated here
/// \param cachePath as far as we need two sides unfold, cache one to reduce allocations
MRMESH_API bool reducePathViaVertex( const Mesh & mesh, const MeshTriPoint & start, VertId v, const MeshTriPoint & end, 
    SurfacePath & outPath, std::vector<Vector2f> & tmp, SurfacePath& cachePath );

/// converts any input surface path into geodesic path (so reduces its length): start-path-end;
/// returns actual number of iterations performed
MRMESH_API int reducePath( const Mesh & mesh, const MeshTriPoint & start, SurfacePath & path, const MeshTriPoint & end, int maxIter = 5 );

/// \}

} // namespace MR
