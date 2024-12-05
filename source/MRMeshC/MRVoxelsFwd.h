#pragma once
#include "MRMeshFwd.h"
#include "MRVector3.h"

MR_EXTERN_C_BEGIN

typedef struct MRFloatGrid MRFloatGrid;

typedef struct MRVdbVolume
{
    MRFloatGrid* data;
    MRVector3i dims;
    MRVector3f voxelSize;
    float min;
    float max;
} MRVdbVolume;

MR_EXTERN_C_END
