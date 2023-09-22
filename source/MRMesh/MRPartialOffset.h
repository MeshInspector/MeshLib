#pragma once
#include "MRMeshFwd.h"
#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "MROffset.h"

namespace MR
{

/// Offsets mesh part by converting it to voxels and back
/// and unite it with original mesh (via boolean)
/// note: only OffsetParameters::Type::Shell will work in this function
MRMESH_API Expected<Mesh, std::string> partialOffsetMesh( const MeshPart& mp, float offset, const OffsetParameters& params = {} );

} //namespace MR

#endif //!__EMSCRIPTEN__
