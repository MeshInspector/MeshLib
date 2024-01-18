#pragma once

#include "MRMeshFwd.h"

namespace MR
{

// inspired by "On the Shape of a Set of Points in the Plane" by HERBERT EDELSBRUNNER, DAVID G. KIRKPATRICK, AND RAIMUND SEIDEL
// https://www.cs.jhu.edu/~misha/Fall13b/Papers/Edelsbrunner93.pdf

/// find all triangles of alpha-shape with negative alpha = -1/radius,
/// where each triangle contains point #v and two other points
MRMESH_API void findAlphaShapeNeiTriangles( const PointCloud & cloud, VertId v, float radius,
    Triangulation & appendTris,  ///< found triangles will be appended here
    std::vector<VertId> & neis,  ///< temporary storage to avoid memory allocations, it will be filled with all neighbours within 2*radius
    bool onlyLargerVids );       ///< if true then two other points must have larger ids (to avoid finding same triangles several times)

/// find all triangles of alpha-shape with negative alpha = -1/radius
[[nodiscard]] MRMESH_API Triangulation findAlphaShapeAllTriangles( const PointCloud & cloud, float radius );

} //namespace MR
