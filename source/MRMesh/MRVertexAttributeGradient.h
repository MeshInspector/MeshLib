#pragma once
#include "MRMeshFwd.h"

namespace MR
{

/// Calculates gradient for each vertex in mesh, based on vertexAttribute
MRMESH_API Vector<Vector3f, VertId> vertexAttributeGradient( const Mesh& mesh, const Vector<float, VertId>& vertexAttribute );

}