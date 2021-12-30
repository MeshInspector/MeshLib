#pragma once
#include "MRMeshFwd.h"
#include "MRVector3.h"

namespace MR
{
// Base is "lower" corner of the cube coordinates
MRMESH_API Mesh makeCube( const Vector3f& size = Vector3f::diagonal(1.0f), const Vector3f& base = Vector3f::diagonal(-0.5f) );

MRMESH_API Mesh makeParallelepiped(const Vector3f side[3], const Vector3f& base);

}
