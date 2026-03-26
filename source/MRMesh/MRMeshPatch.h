#pragma once
#include "MRMeshFwd.h"
#include "MRFillHoleNicely.h"

namespace MR
{

/// removes \param patchBS from \param mesh and fills every hole appeared with new triangles using \ref fillHoleNicely and \params settings;
/// if 'settings.subdivideSettings.maxEdgeLen' <= 0 uses patch boundary average edge length * 1.5f
/// returns new faces
MRMESH_API FaceBitSet patchMesh( Mesh& mesh, const FaceBitSet& patchBS, const FillHoleNicelySettings& settings = {} );

} //namespace MR
