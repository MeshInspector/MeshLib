#pragma once

#include "MRPointsToMeshProjector.h"
#include "MRProgressCallback.h"
#include <optional>

namespace MR
{

/// \addtogroup MeshAlgorithmGroup
/// \{

struct ShrinkwrapParameters : MeshProjectionParameters
{
    /// each vertex is displaced by this distance from the closest point found, in the direction from
    /// that point toward the original position of the vertex, so a positive offset keeps the vertex on
    /// the side of refMesh it came from; the pseudonormal of refMesh is used instead for the vertices
    /// lying exactly on refMesh; measured in the coordinates of refMesh
    float offset = 0;

    /// if provided then only the vertices from this region are moved, and the others remain in place
    const VertBitSet * region = nullptr;

    /// if provided then the projections are computed by it, otherwise CPU's computations are used
    IPointsToMeshProjector * projector = nullptr;
};

/// moves every vertex of the mesh in the closest point on the reference mesh;
/// the vertices having no projection within MeshProjectionParameters::upDistLimitSq remain in place;
/// this function changes vertex coordinates only, keeping mesh topology intact, so the result
/// can self-intersect where refMesh is concave, and a non-zero offset makes it much more likely;
/// consider offsetMesh(...) if a self-intersection free result is required
/// \return false if the operation was canceled by the progress callback
MRMESH_API bool shrinkwrap( Mesh & mesh, const Mesh & refMesh, const ShrinkwrapParameters & params = {},
    const ProgressCallback & cb = {} );

/// computes for every vertex of the mesh its closest point on the reference mesh,
/// leaving the vertices having no projection within MeshProjectionParameters::upDistLimitSq in place;
/// the result is expressed in the coordinates of the mesh being projected
/// \return no value if the operation was canceled by the progress callback
[[nodiscard]] MRMESH_API std::optional<VertCoords> findShrinkwrapPositions( const Mesh & mesh, const Mesh & refMesh,
    const ShrinkwrapParameters & params = {}, const ProgressCallback & cb = {} );

/// \}

} // namespace MR
