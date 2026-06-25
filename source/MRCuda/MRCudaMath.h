#pragma once
#ifndef MR_PARSING_FOR_ANY_BINDINGS

#include "exports.h"

#include "MRMesh/MRMeshFwd.h"

#ifndef __HIP_PLATFORM_AMD__
struct float3;
struct int3;
#else
template<typename T, unsigned int rank>
struct HIP_vector_type;

using float3 = HIP_vector_type<float, 3>;
using int3 = HIP_vector_type<int, 3>;
#endif

namespace MR::Cuda
{

// structs from MRCudaMath.cuh
struct Matrix4;

// copy from CPU to GPU structs
MRCUDA_API float3 fromVec( const Vector3f& v );
MRCUDA_API int3 fromVec( const Vector3i& v );
MRCUDA_API Matrix4 fromXf( const MR::AffineXf3f& xf );

} // namespace MR::Cuda
#endif
