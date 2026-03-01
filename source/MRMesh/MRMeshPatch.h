#pragma once
#include "MRMeshFwd.h"
#include "MRFillHoleNicely.h"

namespace MR
{

/// removes \param patchBS from \param mesh and fills it with new triangulation using \params settings
/// if 'settings.subdivideSettings.maxEdgeLen' <= 0 uses patch boundary average edge length * 1.5f
/// returns new faces
MRMESH_API FaceBitSet patchMesh( Mesh& mesh, const FaceBitSet& patchBS, const FillHoleNicelySettings& settings = {} );

}