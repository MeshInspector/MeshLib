#pragma once

#include "MRMeshFwd.h"

namespace MR
{

// deletes the face, also deletes its edges and vertices if they were not shared with other faces
MRMESH_API void deleteFace( MeshTopology & topology, FaceId f );

// deletes multiple given faces
MRMESH_API void deleteFaces( MeshTopology & topology, const FaceBitSet & fs );

// deletes object faces with normals pointed to the target geometry center
MRMESH_API void deleteTargetFaces( Mesh &obj, const Vector3f &targetCenter );
MRMESH_API void deleteTargetFaces( Mesh &obj, const Mesh & target );

} //namespace MR
