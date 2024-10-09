#pragma once

#include "MRVector3.h"
#include "MRVectorTraits.h"
#include "MRPch/MRBindingMacros.h"
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

    /// if true then maximal displacement of each point during denoising will be limited
    bool limitNearInitial = false;

    /// maximum distance between a point and its position before relaxation, ignored if limitNearInitial = false
    float maxInitialDist = 0;

    // Fixes ABI incompatibility. Without this GCC 12+ warns here with `-Wabi=16`.
    // Read our `cmake/Modules/CompilerOptions.cmake` (the part about `-Wabi=16`) for details.
    // This is enabled for GCC 11 and older because they're buggy, and for GCC 12 because for it we enable the warning to catch other similar cases.
    // For other compilers it's disabled for clarity, but should have no effect on struct layout.
    MR_BIND_IGNORE int _padding;
};

enum class RelaxApproxType
{
    Planar,
    Quadric,
};

/// if (pos) is within the ball with the center at (guidePos) and squared radius (maxGuideDistSq), then returns (pos);
/// otherwise returns the point on the ball's border closest to (pos)
template <typename V>
[[nodiscard]] inline V getLimitedPos( const V & pos, const V & guidePos, typename VectorTraits<V>::BaseType maxGuideDistSq )
{
    assert( maxGuideDistSq > 0 );
    const auto d = pos - guidePos;
    float distSq = sqr( d );
    if ( distSq <= maxGuideDistSq )
        return pos;
    return guidePos + std::sqrt( maxGuideDistSq / distSq ) * d;
}

} // namespace MR
