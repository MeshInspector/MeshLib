#pragma once

#include "MRMeshFwd.h"
#include "MRBox.h"

namespace MR
{

/// passes through all region points and finds the minimal bounding box containing all of them;
/// if toWorld transformation is given then returns minimal bounding box in world space
/// \ingroup MathGroup
template<typename V>
Box<V> computeBoundingBox( const Vector<V, VertId> & points, const VertBitSet & region, const AffineXf<V> * toWorld = nullptr );
template<typename V>
Box<V> computeBoundingBox( const Vector<V, VertId>& points, const VertBitSet* region = nullptr, const AffineXf<V>* toWorld = nullptr );

} // namespace MR
