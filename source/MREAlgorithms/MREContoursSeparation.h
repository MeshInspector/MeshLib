#pragma once
#include "exports.h"
#include "MRMesh/MRMeshFwd.h"
#include <functional>

namespace MRE
{
// Callback for each MeshEdgePoint in path
using PathMeshEdgePointCallback = std::function<void( const MR::MeshEdgePoint& mep )>;

// Separates mesh into disconnected by contour components (independent components are not returned),
// faces that are intersected by contour does not belong to any component.
// Calls callback for each MeshEdgePoint in contour respecting order, 
// ignoring MeshTriPoints (if projection of input point lay inside face)
MREALGORITHMS_API std::vector<MR::FaceBitSet> separateClosedContour( const MR::Mesh& mesh, const std::vector<MR::Vector3f>& contour,
    const PathMeshEdgePointCallback& cb = {} );
}
