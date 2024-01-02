#pragma once

#include "MRMeshFwd.h"

namespace MR
{

// inspired by "On the Shape of a Set of Points in the Plane" by HERBERT EDELSBRUNNER, DAVID G. KIRKPATRICK, AND RAIMUND SEIDEL
// https://www.cs.jhu.edu/~misha/Fall13b/Papers/Edelsbrunner93.pdf

/// find all triangles of alpha-shape with negative alpha = -1/radius,
/// where each triangle contains point #v and two other points with larger ids
void findAlphaShapeNeiTriangles( const PointCloud & cloud, VertId v, float radius,
    Triangulation & appendTris,  ///< found triagles will be appended here
    std::vector<VertId> & tmp ); ///< temporary storage to avoid memory allocations

/// find all triangles of alpha-shape with negative alpha = -1/radius
Triangulation findAlphaShapeAllTriangles( const PointCloud & cloud, float radius );

} //namespace MR
