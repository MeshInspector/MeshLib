#pragma once

#include "MRSignDetectionMode.h"
#include <cfloat>

namespace MR
{

/// options determining computation of distance from a point to a mesh
struct DistanceToMeshOptions
{
    /// minimum squared distance from a point to mesh to be computed precisely
    float minDistSq{ 0 };

    /// maximum squared distance from a point to mesh to be computed precisely
    float maxDistSq{ FLT_MAX };

    /// what to do if actual distance is outside [min, max) range:
    /// true - return std::nullopt for std::optional<float> or NaN for float,
    /// false - return approximate value of the distance (with correct sign in case of SignDetectionMode::HoleWindingRule);
    /// please note that in HoleWindingRule the sign can change even for too small or too large distances,
    /// so if you would like to get closed mesh from marching cubes, set false here
    bool nullOutsideMinMax = true;

    /// only for SignDetectionMode::HoleWindingRule:
    /// positive distance if winding number below or equal this threshold;
    /// ideal threshold: 0.5 for closed meshes; 0.0 for planar meshes
    float windingNumberThreshold = 0.5f;

    /// only for SignDetectionMode::HoleWindingRule:
    /// determines the precision of fast approximation: the more the better, minimum value is 1
    float windingNumberBeta = 2;
};

/// options determining computation of signed distance from a point to a mesh
struct SignedDistanceToMeshOptions : DistanceToMeshOptions
{
    /// the method to compute distance sign
    SignDetectionMode signMode{ SignDetectionMode::ProjectionNormal };
};

} //namespace MR
