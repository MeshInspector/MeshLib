#pragma once

#include "exports.h"

#include "MRMesh/MRMeshFwd.h"

struct float3;

namespace MR::Cuda
{

// structs from MRCudaMath.cuh
struct Matrix4;

// copy from CPU to GPU structs
MRCUDA_API float3 fromVec( const Vector3f& v );
MRCUDA_API Matrix4 fromXf( const MR::AffineXf3f& xf );

} // namespace MR::Cuda
