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

#ifndef MRMESH_NO_OPENVDB
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

    [[nodiscard]] size_t heapBytes() const
    {
        // `...IfSupported` because `T` can be a functor (`std::function`), then we can't compute this.
        return MR::heapBytesIfSupported( data );
    }
};

/// represents a box in 3D space subdivided on voxels stored in T;
/// and stores minimum and maximum values among all valid voxels
template <typename T>
struct VoxelsVolumeMinMax : VoxelsVolume<T>
{
    using typename VoxelsVolume<T>::ValueType;
    using VoxelsVolume<T>::data;
    using VoxelsVolume<T>::dims;
    using VoxelsVolume<T>::voxelSize;
    using VoxelsVolume<T>::heapBytes;

    ValueType min = std::numeric_limits<ValueType>::max();
    ValueType max = std::numeric_limits<ValueType>::lowest();
};

} //namespace MR
