#pragma once

#include "MRMeshFwd.h"

namespace MR
{

// adds triangles along the boundary to straighten it;
// new triangle is added only if 
//  1) aspect ratio of the new triangle is at most maxTriAspectRatio,
//  2) dot product of its normal with neighbor triangles is at least minNeiNormalsDot.
MRMESH_API void straightenBoundary( Mesh & mesh, EdgeId bd, float minNeiNormalsDot, float maxTriAspectRatio, FaceBitSet* newFaces = nullptr );

} //namespace MR
