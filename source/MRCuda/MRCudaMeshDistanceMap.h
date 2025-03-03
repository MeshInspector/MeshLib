#pragma once

#include "exports.h"

#include "MRMesh/MRDistanceMap.h"
#include "MRMesh/MRMesh.h"

namespace MR::Cuda
{

/// computes distance (height) map for given projection parameters
/// using float-precision for finding ray-mesh intersections, which is faster but less reliable
MRCUDA_API Expected<DistanceMap> computeDistanceMap( const Mesh& mesh, const MeshToDistanceMapParams& params,
    ProgressCallback cb = {}, std::vector<MeshTriPoint>* outSamples = nullptr );

/// Computes memory consumption of computeDistanceMap function
MRCUDA_API size_t computeDistanceMapHeapBytes( const Mesh& mesh, const MeshToDistanceMapParams& params, bool needOutSamples = false );

} // namespace MR::Cuda
