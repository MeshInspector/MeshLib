#pragma once

#include "MRMeshFwd.h"
#include "MRVector.h"
#include <array>

namespace MR
{

/// \defgroup MeshNormalsGroup Mesh Normals
/// \ingroup MeshAlgorithmGroup
/// \{

struct [[nodiscard]] MeshNormals
{
    FaceNormals faceNormals;
    VertNormals vertNormals;
};

/// returns a vector with face-normal in every element for valid mesh faces
[[nodiscard]] MRMESH_API FaceNormals computePerFaceNormals( const Mesh & mesh );

/// fills buffer with face-normals as Vector4f for valid mesh faces
MRMESH_API void computePerFaceNormals4( const Mesh & mesh, Vector4f* faceNormals, size_t size );

/// returns a vector with vertex normals in every element for valid mesh vertices
[[nodiscard]] MRMESH_API VertNormals computePerVertNormals( const Mesh & mesh );

/// returns a vector with vertex pseudonormals in every element for valid mesh vertices
/// see http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.107.9173&rep=rep1&type=pdf
[[nodiscard]] MRMESH_API VertNormals computePerVertPseudoNormals( const Mesh & mesh );

/// computes both per-face and per-vertex normals more efficiently then just calling both previous functions
[[nodiscard]] MRMESH_API MeshNormals computeMeshNormals( const Mesh & mesh );

/// normals in three corner of a triangle
using TriangleCornerNormals = std::array<Vector3f, 3>;
/// returns a vector with corner normals in every element for valid mesh faces;
/// corner normals of adjacent triangles are equal, unless they are separated by crease edges
[[nodiscard]] MRMESH_API Vector<TriangleCornerNormals, FaceId> computePerCornerNormals( const Mesh & mesh, const UndirectedEdgeBitSet* creases );

/// \}

} // namespace MR
