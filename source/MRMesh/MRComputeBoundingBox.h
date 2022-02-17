#pragma once

#include "MRMeshFwd.h"

namespace MR
{

// passes through all region points and finds the minimal bounding box containing all of them;
// if toWorld transformation is given then returns minimal bounding box in world space
MRMESH_API Box3f computeBoundingBox( const VertCoords & points, const VertBitSet & region, const AffineXf3f * toWorld = nullptr );

} //namespace MR
