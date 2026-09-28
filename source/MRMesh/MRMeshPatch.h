#pragma once
#include "MRMeshFwd.h"
#include "MRFillHoleNicely.h"

namespace MR
{

/// removes \param patchBS from \param mesh and fills every hole appeared with new triangles using \ref fillHoleNicely and \params settings;
/// if 'settings.subdivideSettings.maxEdgeLen' <= 0 uses patch boundary average edge length * 1.5f
/// returns new faces
MRMESH_API FaceBitSet patchMesh( Mesh& mesh, const FaceBitSet& patchBS, const FillHoleNicelySettings& settings = {} );

/// splits given faces on groups separated by sharp edges (see MeshComponents::getAllComponentsMapBySharpEdges),
/// then removes each group and fills the appeared holes separately using \ref patchMesh;
/// if patch subdivision splits a face of a not yet patched group or of an earlier patch, the new half is added to that group or to the result
/// \return all new faces
MRMESH_API FaceBitSet patchMeshByGroups( Mesh& mesh, const FaceBitSet& faces, float angleThreshold = 0.5f, const FillHoleNicelySettings& settings = {} );

} //namespace MR
