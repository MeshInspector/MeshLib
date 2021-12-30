#pragma once

#include "MRMeshFwd.h"

namespace MR
{

// given two contours:
// 1) of equal size;
// 2) all edges of c0 have no left faces;
// 3) all edges of c1 have no right faces;
// merge the surface along corresponding edges of two contours, and deletes all vertices and edges from c1
MRMESH_API void stitchContours( MeshTopology & topology, const std::vector<EdgeId> & c0, const std::vector<EdgeId> & c1 );

} //namespace MR
