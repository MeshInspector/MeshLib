#pragma once

#include "MRMeshFwd.h"
#include "MRVector.h"
#include <array>

namespace MR
{

/// \defgroup MeshNormalsGroup Mesh Normals
/// \ingroup MeshAlgorithmGroup
/// \{

using FaceNormals = Vector<Vector3f, FaceId>;
using VertexNormals = Vector<Vector3f, VertId>;

struct [[nodiscard]] MeshNormals
{
    FaceNormals faceNormals;
    VertexNormals vertNormals;
};

/// returns a vector with face-normal in every element for valid mesh faces
[[nodiscard]] MRMESH_API FaceNormals computePerFaceNormals( const Mesh & mesh );

/// returns a buffer with face-normals as Vector4f for valid mesh faces
[[nodiscard]] MRMESH_API Buffer<Vector4f> computePerFaceNormals4( const Mesh & mesh, size_t bufferSize = 0 );

/// returns a vector with vert-normal in every element for valid mesh vertices
[[nodiscard]] MRMESH_API VertexNormals computePerVertNormals( const Mesh & mesh );

/// computes both per-face and per-vertex normals more efficiently then just calling both previous functions
[[nodiscard]] MRMESH_API MeshNormals computeMeshNormals( const Mesh & mesh );

/// normals in three corner of a triangle
using TriangleCornerNormals = std::array<Vector3f, 3>;
/// returns a vector with corner normals in every element for valid mesh faces;
/// corner normals of adjacent triangles are equal, unless they are separated by crease edges
[[nodiscard]] MRMESH_API Vector<TriangleCornerNormals, FaceId> computePerCornerNormals( const Mesh & mesh, const UndirectedEdgeBitSet* creases );

/// \}

} // namespace MR
