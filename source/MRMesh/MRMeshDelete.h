#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// \defgroup MeshDeleteGroup Mesh Delete
/// \ingroup MeshAlgorithmGroup
/// \{

/// deletes the face, also deletes its edges and vertices if they were not shared with other faces;
/// deprecated: please call topology.deleteFace instead
[[deprecated]] MRMESH_API void deleteFace( MeshTopology & topology, FaceId f );

/// deletes multiple given faces;
/// deprecated: please call topology.deleteFaces instead
[[deprecated]] MRMESH_API void deleteFaces( MeshTopology & topology, const FaceBitSet & fs );

/// deletes object faces with normals pointed to the target geometry center
MRMESH_API void deleteTargetFaces( Mesh &obj, const Vector3f &targetCenter );
MRMESH_API void deleteTargetFaces( Mesh &obj, const Mesh & target );

/// \}

} // namespace MR
