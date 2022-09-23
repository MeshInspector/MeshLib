#pragma once
#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "MROffset.h"

namespace MR
{

/// Offsets mesh part by converting it to voxels and back
/// and unite it with original mesh (via boolean)
MRMESH_API tl::expected<Mesh, std::string> partialOffsetMesh( const MeshPart& mp, float offset, const OffsetParameters& params = {} );

} //namespace MR

#endif //!__EMSCRIPTEN__
