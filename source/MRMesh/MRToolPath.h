#pragma once
#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "exports.h"
#include "MRMeshFwd.h"
#include "MRPolyline.h"


namespace MR
{

MRMESH_API std::shared_ptr<Polyline3> getToolPath( Mesh& mesh, float millRadius, float voxelSize, float sectionStep, float critLength = 0.1f );

}
#endif
