#pragma once
#include "MRVector3.h"
#include "MRHeapBytes.h"
#include <limits>
#include <vector>

namespace MR
{

template <typename T>
struct VoxelTraits;

template <typename T>
struct VoxelTraits<std::vector<T>>
{
    using ValueType = T;
};

template <typename T>
struct VoxelTraits<VoxelValueGetter<T>>
{
    using ValueType = T;
};

#ifndef MRMESH_NO_VOXEL
template <>
struct VoxelTraits<FloatGrid>
{
    using ValueType = float;
};
#endif

/// represents a box in 3D space subdivided on voxels stored in T
template <typename T>
struct VoxelsVolume
{
    using ValueType = typename VoxelTraits<T>::ValueType;

    T data;
    Vector3i dims;
    Vector3f voxelSize{ 1.f, 1.f, 1.f };
    ValueType min = std::numeric_limits<ValueType>::max();
    ValueType max = std::numeric_limits<ValueType>::lowest();

    [[nodiscard]] size_t heapBytes() const { return MR::heapBytes( data ); }
};

} //namespace MR
