#pragma once
#include "MRMeshFwd.h"
#include "MRVector3.h"
#include "MRVector2.h"
#include "MRExpected.h"

namespace MR
{

/// structure with parameters for `compensateRadius` function
struct CompensateRadiusParams
{
    /// Z direction of milling tool
    Vector3f direction;

    ///  radius of spherical tool
    float toolRadius{ 0.0f };

    /// region of the mesh that will be compensated
    /// it should not contain closed components
    /// also please note that boundaries of the region are fixed
    const FaceBitSet* region{ nullptr };

    /// maximum iteration of applying algorithm (each iteration improves result a little bit)
    int maxIterations{ 100 };

    /// how many hops to expand around each moved vertex for relaxation
    int relaxExpansion = 3;

    /// how many iterations of relax is applied on each compensation iteration
    int relaxIterations = 5;

    /// force of relaxations on each compensation iteration
    float relaxForce = 0.3f;

    ProgressCallback callback;
};

/// compensate spherical milling tool radius in given mesh region making it possible to mill it
/// note that tool milling outer surface of the mesh
/// also please note that boundaries of the region are fixed
[[nodiscard]] MRMESH_API Expected<void> compensateRadius( Mesh& mesh, const CompensateRadiusParams& params );

}
