#pragma once
#include "exports.h"
#include "MRMesh/MRMesh.h"

namespace MR { namespace Cuda {

/// Computes distance of 2d contours according to ContourToDistanceMapParams (works correctly only when withSign==false)
MRCUDA_API std::vector<MR::MeshProjectionResult> findProjections( const std::vector<Vector3f>& points, const MR::Mesh& mesh, float upDistLimitSq = FLT_MAX, float loDistLimitSq = 0);

}}