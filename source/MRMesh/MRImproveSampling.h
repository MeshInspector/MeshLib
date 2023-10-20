#pragma once

#include "MRMeshFwd.h"
#include "MRProgressCallback.h"

namespace MR
{

/// Finds more representative sampling starting from a given one following k-means method;
/// \param samples input and output selected sample points from \param cloud;
/// \param numIters the number of algorithm iterations to perform;
/// \return false if it was terminated by the callback
/// \ingroup PointCloudGroup
MRMESH_API bool improveSampling( const PointCloud & cloud, VertBitSet & samples, int numIters, const ProgressCallback & cb = {} );

} //namespace MR
