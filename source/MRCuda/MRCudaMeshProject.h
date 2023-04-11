#pragma once
#include "exports.h"
#include "MRMesh/MRMesh.h"

namespace MR { namespace Cuda {

/// Computes distance of 2d contours according to ContourToDistanceMapParams
MRCUDA_API std::vector<MR::MeshProjectionResult> findProjections( const std::vector<Vector3f>& points, const MR::Mesh& mesh, const AffineXf3f* xf = nullptr, const AffineXf3f* refXfPtr = nullptr, float upDistLimitSq = FLT_MAX, float loDistLimitSq = 0);

}}