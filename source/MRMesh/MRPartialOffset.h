#pragma once
#ifndef __EMSCRIPTEN__
#include "MROffset.h"

namespace MR
{

/// Offsets mesh part by converting it to voxels and back
/// and unite it with original mesh (via boolean)
MRMESH_API Mesh partialOffsetMesh( const MeshPart& mp, float offset, const OffsetParameters& params = {} );

} //namespace MR

#endif //!__EMSCRIPTEN__
