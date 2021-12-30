#pragma once

#include "MRMeshFwd.h"

namespace MR
{

struct RelaxParams
{
    // number of iterations
    int iterations{ 1 };
    // region to relax
    const VertBitSet* region{ nullptr };
    // speed of relaxing, typical values (0.0, 0.5]
    float force{ 0.5f };
};

// applies given number of relaxation iterations to the whole mesh ( or some region if it is specified )
MRMESH_API void relax( Mesh& mesh, const RelaxParams params = {} );

// applies given number of relaxation iterations to the whole mesh ( or some region if it is specified )
// do not really keeps volume but tries hard
MRMESH_API void relaxKeepVolume( Mesh& mesh, const RelaxParams params = {} );

enum class RelaxApproxType 
{
    Planar,
    Quadric
};

struct MeshApproxRelaxParams : RelaxParams
{
    // radius to find neighbors by surface
    // 0.0f - default = 1e-3 * sqrt(surface area)
    float surfaceDilateRadius{ 0.0f };
    RelaxApproxType type{ RelaxApproxType::Planar };
};

// applies given number of relaxation iterations to the whole mesh ( or some region if it is specified )
// approx neighborhoods
MRMESH_API void relaxApprox( Mesh& mesh, const MeshApproxRelaxParams params = {} );

// applies at most given number of relaxation iterations the spikes detected by given threshold
MRMESH_API void removeSpikes( Mesh & mesh, int maxIterations, float minSumAngle, const VertBitSet * region = nullptr );

}
