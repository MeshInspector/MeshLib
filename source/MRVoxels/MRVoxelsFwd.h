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

class ObjectVoxels;

struct OpenVdbFloatGrid;
using FloatGrid = std::shared_ptr<OpenVdbFloatGrid>;

MR_CANONICAL_TYPEDEFS( (template <typename T> struct), VoxelsVolumeMinMax,
    ( SimpleVolumeMinMax, VoxelsVolumeMinMax<std::vector<float>> )
    ( SimpleVolumeMinMaxU16, VoxelsVolumeMinMax<std::vector<uint16_t>> )
    ( VdbVolume, VoxelsVolumeMinMax<FloatGrid> )
)

template <typename T>
using VoxelValueGetter = std::function<T ( const Vector3i& )>;

MR_CANONICAL_TYPEDEFS( (template <typename T> struct), VoxelsVolume,
    ( FunctionVolume, VoxelsVolume<VoxelValueGetter<float>> )
    ( FunctionVolumeU8, VoxelsVolume<VoxelValueGetter<uint8_t>> )
    ( SimpleVolume, VoxelsVolume<std::vector<float>> )
    ( SimpleVolumeU16, VoxelsVolume<std::vector<uint16_t>> )
)

} // namespace MR
