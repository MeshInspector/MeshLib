#pragma once

#include "MRMeshFwd.h"
#include "MRProgressCallback.h"
#include <cfloat>

namespace MR
{

/// Modifies \p mesh shifting each vertex along its pseudonormal by the corresponding \p offset
/// @return false if cancelled.
MRMESH_API bool offsetVerts( Mesh& mesh, const VertMetric& offset, const ProgressCallback& cb = {} );

struct ThickenParams
{
    /// the amount of offset for original mesh vertices
    float outsideOffset = 0;

    /// the amount of offset for cloned mirrored mesh vertices in the opposite direction
    float insideOffset = 0;
};

/// given a mesh \p m, representing a surface,
/// creates new closed mesh by cloning mirrored mesh, and shifting original part and cloned part in different directions according to \p params,
/// if original mesh was open then stitches corresponding boundaries of two parts
MRMESH_API Mesh makeThickMesh( const Mesh & m, const ThickenParams & params );

} //namespace MR
