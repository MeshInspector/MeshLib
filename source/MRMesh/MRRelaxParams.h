#pragma once
#include "MRMeshFwd.h"

namespace MR
{

struct RelaxParams
{
    /// number of iterations
    int iterations{1};
    /// region to relax
    const VertBitSet *region{nullptr};
    /// speed of relaxing, typical values (0.0, 0.5]
    float force{0.5f};
};

enum class RelaxApproxType
{
    Planar,
    Quadric,
};

} // namespace MR