#pragma once

#include "MRMeshFwd.h"
#include "MRProgressCallback.h"

namespace MR
{

struct ImproveSamplingSettings
{
    /// the number of algorithm iterations to perform
    int numIters = 1;

    /// if a sample represents less than this number of input points then such sample will be discarded;
    /// it can be used to remove outliers
    int minPointsInSample = 1;

    /// optional output: mapping from input point id to sample id
    VertMap * pt2sm = nullptr;

    /// optional output: new cloud containing averaged points and normals for each sample
    PointCloud * cloudOfSamples = nullptr;

    /// optional output: the number of points in each sample
    Vector<int, VertId> * ptsInSm = nullptr;

    /// optional input: colors of input points
    const VertColors * ptColors = nullptr;

    /// optional output: averaged colors of samples
    VertColors * smColors = nullptr;

    /// output progress status and receive cancel signal
    ProgressCallback progress;
};

/// Finds more representative sampling starting from a given one following k-means method;
/// \param samples input and output selected sample points from \param cloud;
/// \return false if it was terminated by the callback
/// \ingroup PointCloudGroup
MRMESH_API bool improveSampling( const PointCloud & cloud, VertBitSet & samples, const ImproveSamplingSettings & settings );

} //namespace MR
