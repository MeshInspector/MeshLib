#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// given a mesh \p m, representing a surface,
/// creates new closed mesh by cloning mirrored mesh, and shifting original part and cloned part in different direction on \p halfWidth each,
/// if original mesh was open then stitches corresponding boundaries of two parts
MRMESH_API Mesh makeThickMesh( const Mesh & m, float halfWidth );

} //namespace MR
