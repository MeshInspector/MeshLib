#pragma once
#include "exports.h"
#include "MRMesh/MROffset.h"

namespace MRE
{
using namespace MR;
/// Offsets mesh part by converting it to voxels and back
/// and unite it with original mesh (via boolean)
MREALGORITHMS_API Mesh partialOffsetMesh( const MeshPart& mp, float offset, const OffsetParameters& params = {} );

}