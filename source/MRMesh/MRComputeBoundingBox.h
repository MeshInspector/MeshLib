#pragma once

#include "MRMeshFwd.h"

namespace MR
{

// passes through all region points and finds the minimal bounding box containing all of them;
// if toWorld transformation is given then returns minimal bounding box in world space
template<typename V>
MRMESH_API Box<V> computeBoundingBox( const Vector<V, VertId> & points, const VertBitSet & region, const AffineXf<V> * toWorld = nullptr );

} //namespace MR
