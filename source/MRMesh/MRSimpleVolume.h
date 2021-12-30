#pragma once

#include "MRVector3.h"
#include <cfloat>
#include <vector>

namespace MR
{
struct SimpleVolume
{
    std::vector<float> data;
    Vector3i dims;
    Vector3f voxelSize;
    float min = FLT_MAX;
    float max = -FLT_MAX;
};
}