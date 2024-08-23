#pragma once

#include "MRMeshFwd.h"
#include "MROffset.h"

namespace MR
{

/// Offsets mesh part by converting it to voxels and back
/// and unite it with original mesh (via boolean)
/// note: only OffsetParameters.signDetectionMode = SignDetectionMode::Unsigned will work in this function
MRMESH_API Expected<Mesh> partialOffsetMesh( const MeshPart& mp, float offset, const GeneralOffsetParameters& params = {} );

} //namespace MR
