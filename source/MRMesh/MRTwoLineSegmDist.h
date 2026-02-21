#pragma once

#include "MRVector3.h"
#include "MRLineSegm.h"

namespace MR
{

template<class T>
struct TwoLineSegmClosestPoints
{
    /// the closest points each from its respective segment
    Vector3<T> a, b;

    // if both closest points are in segment endpoints, then directed from closest point a to closest point b,
    // if both closest points are inner to the segments, then its orthogonal to both segments and directed from a to b,
    // otherwise it is orthogonal to the segment with inner closest point and rotated toward/away the other closest point in endpoint
    Vector3<T> dir;
};
using TwoLineSegmClosestPointsf = TwoLineSegmClosestPoints<float>;
using TwoLineSegmClosestPointsd = TwoLineSegmClosestPoints<double>;

/// computes the closest points on two line segments
[[nodiscard]] MRMESH_API TwoLineSegmClosestPointsf findTwoLineSegmClosestPoints( const LineSegm3f& a, const LineSegm3f& b );
[[nodiscard]] MRMESH_API TwoLineSegmClosestPointsd findTwoLineSegmClosestPoints( const LineSegm3d& a, const LineSegm3d& b );

} // namespace MR
