#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// returns the region located to the left from the given edges;
/// if the contour(s) are not separating and there is a tunnel connecting left and right side of the contour(s),
/// then all faces from contour's connected components are returned;
/// please use \ref fillContourLeftByGraphCut for not-separating contours instead
MRMESH_API FaceBitSet fillContourLeft( const MeshTopology & topology, const EdgePath & contour );
MRMESH_API FaceBitSet fillContourLeft( const MeshTopology & topology, const std::vector<EdgePath> & contours );

} //namespace MR
