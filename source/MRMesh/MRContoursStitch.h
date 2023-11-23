#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// given two contours:
/// 1) of equal size;
/// 2) all edges of c0 have no left faces;
/// 3) all edges of c1 have no right faces;
/// merge the surface along corresponding edges of two contours, and deletes all vertices and edges from c1
MRMESH_API void stitchContours( MeshTopology & topology, const EdgePath & c0, const EdgePath & c1 );

/// given a closed loop of edges, splits the surface along that loop so that after return:
/// 1) returned loop has the same size as input, with corresponding edges in same indexed elements of both;
/// 2) all edges of c0 have no left faces;
/// 3) all returned edges have no right faces;
MRMESH_API EdgeLoop cutAlongEdgeLoop( MeshTopology & topology, const EdgeLoop & c0 );

/// given a closed loop of edges, splits the surface along that loop so that after return:
/// 1) returned loop has the same size as input, with corresponding edges in same indexed elements of both;
/// 2) all edges of c0 have no left faces;
/// 3) all returned edges have no right faces;
/// 4) vertices of the given mesh are updated
MRMESH_API EdgeLoop cutAlongEdgeLoop( Mesh& mesh, const EdgeLoop& c0 );

} //namespace MR
