#pragma once

#include "exports.h"

#include "config.h"

#include "MRMeshFwd.h"

namespace MR
{

template <typename T>
struct VoxelsVolume;

template <typename T>
struct VoxelsVolumeMinMax;

using SimpleVolume = VoxelsVolumeMinMax<std::vector<float>>;
using SimpleVolumeU16 = VoxelsVolumeMinMax<std::vector<uint16_t>>;

template <typename T>
using VoxelValueGetter = std::function<T( const Vector3i& )>;
using FunctionVolume = VoxelsVolume<VoxelValueGetter<float>>;
using FunctionVolumeU8 = VoxelsVolume<VoxelValueGetter<uint8_t>>;

#ifndef MRMESH_NO_OPENVDB
class ObjectVoxels;

struct OpenVdbFloatGrid;
using FloatGrid = std::shared_ptr<OpenVdbFloatGrid>;
using VdbVolume = VoxelsVolumeMinMax<FloatGrid>;
#endif

} // namespace MR
