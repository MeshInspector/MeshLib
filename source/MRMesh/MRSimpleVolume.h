#pragma once
#include "MRVector3.h"
#include "MRHeapBytes.h"
#include <cfloat>
#include <vector>

namespace MR
{

template <typename T>
struct VoxelsVolume
{
    T data;
    Vector3i dims;
    Vector3f voxelSize{ 1.f, 1.f, 1.f };
    float min = FLT_MAX;
    float max = -FLT_MAX;

    [[nodiscard]] size_t heapBytes() const { return MR::heapBytes( data ); }
};

}