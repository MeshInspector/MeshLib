#pragma once

#include "MRVoxelsFwd.h"
#include "MRFloatGrid.h"

#include "MRMesh/MRVector3.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRHeapBytes.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MRVector.h"
#include "MRMesh/MRBitSet.h"

#include <limits>

namespace MR
{

template <typename T>
struct VoxelTraits;

template <typename T>
struct VoxelTraits<Vector<T, VoxelId>>
{
    using ValueType = T;
};

template <>
struct VoxelTraits<VoxelBitSet>
{
    using ValueType = bool;
};

template <typename T>
struct VoxelTraits<VoxelValueGetter<T>>
{
    using ValueType = T;
};

template <>
struct VoxelTraits<FloatGrid>
{
    using ValueType = float;
};

/// represents a box in 3D space subdivided on voxels stored in T
template <typename T>
struct VoxelsVolume
{
    using ValueType = typename VoxelTraits<T>::ValueType;

    T data;
    Vector3i dims;
    Vector3f voxelSize{ 1.f, 1.f, 1.f };

    [[nodiscard]] size_t heapBytes() const { return MR::heapBytes( data ); }
};

/// represents a box in 3D space subdivided on voxels stored in T;
/// and stores minimum and maximum values among all valid voxels
template <typename T>
struct VoxelsVolumeMinMax : VoxelsVolume<T>, MinMax<typename VoxelsVolume<T>::ValueType>
{
    using typename VoxelsVolume<T>::ValueType;
    using VoxelsVolume<T>::data;
    using VoxelsVolume<T>::dims;
    using VoxelsVolume<T>::voxelSize;
    using VoxelsVolume<T>::heapBytes;
};


/// converts function volume into simple volume
MRVOXELS_API Expected<SimpleVolumeMinMax> functionVolumeToSimpleVolume( const FunctionVolume& volume, const ProgressCallback& callback = {} );

} //namespace MR
