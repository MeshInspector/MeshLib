#pragma once

#include "MRMeshFwd.h"
#include "MRProgressCallback.h"

namespace MR
{

/// Parameters structure for enweightning points in \ref MR::improveSampling function
struct ImproveSamplingCurvatureMode
{
    enum Type
    {
        None, ///< use uniform weights
        Linear ///< point weight rise up linear with curvature coefficients
    } type{ Type::None };
    /// radius of curvature calculation, if <= 0 use twice distance between first two samples
    float radius{ 0.0f };
};

/// Finds more representative sampling starting from a given one following k-means method;
/// \param samples input and output selected sample points from \param cloud;
/// \param numIters the number of algorithm iterations to perform;
/// \param curvatureWeights use curvature to have more samples in highly curved areas
/// \return false if it was terminated by the callback
/// \ingroup PointCloudGroup
MRMESH_API bool improveSampling( const PointCloud& cloud, VertBitSet& samples, int numIters, 
    const ImproveSamplingCurvatureMode& curvatureMode = {},
    const ProgressCallback& cb = {} );

} //namespace MR
