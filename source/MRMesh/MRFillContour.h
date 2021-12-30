#pragma once

#include "MRMeshFwd.h"

namespace MR
{

// fill region located to the left from given edges
MRMESH_API FaceBitSet fillContourLeft( const MeshTopology & topology, const EdgePath & contour );
MRMESH_API FaceBitSet fillContourLeft( const MeshTopology & topology, const std::vector<EdgePath> & contours );

} //namespace MRE
