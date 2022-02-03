#pragma once
#ifndef __EMSCRIPTEN__
#include "MRMeshFwd.h"
#include <memory>

namespace MR
{

// Computes summary volume of given meshes converting it to voxels of given size
// note that each mesh should have closed topology
// speed and precision depends on voxelSize (smaller voxel - faster, less precise; bigger voxel - slower, more precise)
MRMESH_API float voxelizeAndComputeVolume( const std::vector<std::shared_ptr<Mesh>>& meshes, const AffineXf3f& xf, const Vector3f& voxelSize );

}
#endif
