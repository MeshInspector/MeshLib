#pragma once

#include "config.h"

#ifdef _WIN32
#   ifdef MRVOXELS_EXPORT
#       define MRVOXELS_API __declspec(dllexport)
#   else
#       define MRVOXELS_API __declspec(dllimport)
#   endif
#   define MRVOXELS_CLASS
#else
#   define MRVOXELS_API   __attribute__((visibility("default")))
#   define MRVOXELS_CLASS __attribute__((visibility("default")))
#endif

#include <MRMesh/MRMeshFwd.h>

namespace MR
{

template <typename T>
struct VoxelsVolume;

template <typename T>
struct VoxelsVolumeMinMax;

using SimpleVolume = VoxelsVolumeMinMax<std::vector<float>>;
using SimpleVolumeU16 = VoxelsVolumeMinMax<std::vector<uint16_t>>;

template <typename T>
using VoxelValueGetter = std::function<T ( const Vector3i& )>;
using FunctionVolume = VoxelsVolume<VoxelValueGetter<float>>;
using FunctionVolumeU8 = VoxelsVolume<VoxelValueGetter<uint8_t>>;

class ObjectVoxels;

struct OpenVdbFloatGrid;
using FloatGrid = std::shared_ptr<OpenVdbFloatGrid>;
using VdbVolume = VoxelsVolumeMinMax<FloatGrid>;

} // namespace MR
