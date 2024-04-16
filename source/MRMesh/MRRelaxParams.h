#pragma once

#include "MRVector3.h"
#include <cassert>

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

    /// guide positions of all points
    const VertCoords *guidePos = nullptr;

    /// maximum squared distance between a point and its guide position, ignored if guidePos=nullptr
    float maxGuideDistSq = 1;
};

enum class RelaxApproxType
{
    Planar,
    Quadric,
};

/// of pos is within the ball with the center at guidePos and squared radius maxGuideDistSq, then returns pos;
/// otherwise returns the point on the ball border closest to pos
[[nodiscard]] inline Vector3f getLimitedPos( const Vector3f & pos, const Vector3f & guidePos, float maxGuideDistSq )
{
    assert( maxGuideDistSq > 0 );
    const auto d = pos - guidePos;
    float distSq = d.lengthSq();
    if ( distSq <= maxGuideDistSq )
        return pos;
    return guidePos + std::sqrt( distSq / maxGuideDistSq ) * d;
}

} // namespace MR
