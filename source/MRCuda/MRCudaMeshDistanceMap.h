#pragma once
#include "exports.h"
#include "MRMesh/MRDistanceMap.h"
#include "MRMesh/MRMesh.h"

namespace MR
{
namespace Cuda
{
/// computes distance (height) map for given projection parameters
/// using float-precision for finding ray-mesh intersections, which is faster but less reliable
MRCUDA_API DistanceMap computeDistanceMap( const MR::Mesh& mesh, const MR::MeshToDistanceMapParams& params, 
    ProgressCallback cb = {}, std::vector<MR::MeshTriPoint>* outSamples = nullptr );

/// Computes memory consumption of computeDistanceMap function
MRCUDA_API size_t computeDistanceMapHeapBytes( const MR::Mesh& mesh, const MR::MeshToDistanceMapParams& params, bool needOutSamples = false );
}
}