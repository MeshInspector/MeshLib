#pragma once

#include "MRVector3.h"
#include <cassert>
#include <cfloat>

namespace MR
{

struct RelaxParams
{
    /// number of iterations
    int iterations = 1;

    /// region to relax
    const VertBitSet *region = nullptr;

    /// speed of relaxing, typical values (0.0, 0.5]
    float force = 0.5f;

    /// maximum squared distance between a point and its position before relaxation, ignored if FLT_MAX
    float maxGuideDistSq = FLT_MAX;

    [[nodiscard]] bool limitNearGuide() const { return maxGuideDistSq < FLT_MAX; }
};

enum class RelaxApproxType
{
    Planar,
    Quadric,
};

/// if (pos) is within the ball with the center at (guidePos) and squared radius (maxGuideDistSq), then returns (pos);
/// otherwise returns the point on the ball's border closest to (pos)
template <typename V>
[[nodiscard]] inline V getLimitedPos( const V & pos, const V & guidePos, typename V::ValueType maxGuideDistSq )
{
    assert( maxGuideDistSq > 0 );
    const auto d = pos - guidePos;
    float distSq = d.lengthSq();
    if ( distSq <= maxGuideDistSq )
        return pos;
    return guidePos + std::sqrt( distSq / maxGuideDistSq ) * d;
}

} // namespace MR
