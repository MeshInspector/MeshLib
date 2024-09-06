#pragma once

#include "MRMeshFwd.h"
#include "MRBox.h"

namespace MR
{

/// passes through all region points 
/// 1) in the range [firstVert, lastVert)
/// 2) corresponding to set bits in region (if provided)
/// and finds the minimal bounding box containing all of them;
/// if toWorld transformation is given then returns minimal bounding box in world space
/// \ingroup MathGroup
template<typename V>
Box<V> computeBoundingBox( const Vector<V, VertId> & points, VertId firstVert, VertId lastVert, const VertBitSet* region = nullptr, const AffineXf<V>* toWorld = nullptr );

/// passes through all region points corresponding to set bits in region (if provided)
/// and finds the minimal bounding box containing all of them;
/// if toWorld transformation is given then returns minimal bounding box in world space
/// \ingroup MathGroup
template<typename V>
inline Box<V> computeBoundingBox( const Vector<V, VertId> & points, const VertBitSet* region = nullptr, const AffineXf<V>* toWorld = nullptr )
{
    return computeBoundingBox( points, points.beginId(), points.endId(), region, toWorld );
}

/// passes through all region points corresponding to set bits in region
/// and finds the minimal bounding box containing all of them;
/// if toWorld transformation is given then returns minimal bounding box in world space
/// \ingroup MathGroup
template<typename V>
inline Box<V> computeBoundingBox( const Vector<V, VertId> & points, const VertBitSet & region, const AffineXf<V> * toWorld = nullptr )
{
    return computeBoundingBox( points, points.beginId(), points.endId(), &region, toWorld );
}

} // namespace MR
