#pragma once

#include "MRMesh/MRVector3.h"
#include "MRMesh/MRProgressCallback.h"

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
};

} //namespace MR
