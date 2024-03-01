#pragma once

#include "MRVector3.h"
#include "MRProgressCallback.h"

namespace MR
{

struct DistanceVolumeParams
{
    /// origin point of voxels box
    Vector3f origin;
    /// progress callback
    ProgressCallback cb;
    /// size of voxel on each axis
    Vector3f voxelSize{ 1.0f,1.0f,1.0f };
    /// num voxels along each axis
    Vector3i dimensions{ 100,100,100 };
    /// whether to precompute minimum and maximum values
    /// (requires to iterate through all voxels, which might be computationally expensive for FunctionVolume)
    bool precomputeMinMax = true;
};

} //namespace MR
