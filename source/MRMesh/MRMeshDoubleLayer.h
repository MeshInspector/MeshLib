#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// given a double-layer mesh with one layer having normals outside and the other layer - inside,
/// finds all faces of the outer layer;
/// the algorithm first detects some seed faces of each layer by casting a ray from triangle's center in both directions along the normal;
/// then remaining faces are redistributed toward the closest seed face
[[nodiscard]] MRMESH_API FaceBitSet findOuterLayer( const Mesh & mesh );

} //namespace MR
