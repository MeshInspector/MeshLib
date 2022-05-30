#pragma once

#include "MRMeshFwd.h"
#include <functional>

namespace MR
{

// Callback for each MeshEdgePoint in path
using PathMeshEdgePointCallback = std::function<void( const MeshEdgePoint& mep )>;

// Separates mesh into disconnected by contour components (independent components are not returned),
// faces that are intersected by contour does not belong to any component.
// Calls callback for each MeshEdgePoint in contour respecting order, 
// ignoring MeshTriPoints (if projection of input point lay inside face)
MRMESH_API std::vector<FaceBitSet> separateClosedContour( const Mesh& mesh, const std::vector<Vector3f>& contour,
    const PathMeshEdgePointCallback& cb = {} );

} //namespace MR

