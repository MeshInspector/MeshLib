#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// \defgroup MeshDeleteGroup Mesh Delete
/// \ingroup MeshAlgorithmGroup
/// \{

/// deletes object faces with normals pointed to the target geometry center
MRMESH_API void deleteTargetFaces( Mesh &obj, const Vector3f &targetCenter );
MRMESH_API void deleteTargetFaces( Mesh &obj, const Mesh & target );

/// \}

} // namespace MR
