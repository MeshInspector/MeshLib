#pragma once
#include "MRMeshFwd.h"
#include "MRVector3.h"

MR_EXTERN_C_BEGIN

typedef struct MRFloatGrid MRFloatGrid;

/// represents a box in 3D space subdivided on voxels stored in data;
/// and stores minimum and maximum values among all valid voxels
typedef struct MRVdbVolume
{
    MRFloatGrid* data;
    MRVector3i dims;
    MRVector3f voxelSize;
    float min;
    float max;
} MRVdbVolume;

MR_EXTERN_C_END
