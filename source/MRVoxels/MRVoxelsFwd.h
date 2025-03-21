#pragma once

#include "config.h"

#ifdef _WIN32
#   ifdef MRVoxels_EXPORTS
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

struct MRVOXELS_CLASS OpenVdbFloatGrid;
using FloatGrid = std::shared_ptr<OpenVdbFloatGrid>;

MR_CANONICAL_TYPEDEFS( (template <typename T> struct), MRVOXELS_CLASS VoxelsVolumeMinMax,
    ( SimpleVolumeMinMax, VoxelsVolumeMinMax<std::vector<float>> )
    ( SimpleVolumeMinMaxU16, VoxelsVolumeMinMax<std::vector<uint16_t>> )
    ( VdbVolume, VoxelsVolumeMinMax<FloatGrid> )
)

using VdbVolumes = std::vector<VdbVolume>;

template <typename T>
using VoxelValueGetter = std::function<T ( const Vector3i& )>;

MR_CANONICAL_TYPEDEFS( (template <typename T> struct), MRVOXELS_CLASS VoxelsVolume,
    ( FunctionVolume, VoxelsVolume<VoxelValueGetter<float>> )
    ( FunctionVolumeU8, VoxelsVolume<VoxelValueGetter<uint8_t>> )
    ( SimpleVolume, VoxelsVolume<std::vector<float>> )
    ( SimpleVolumeU16, VoxelsVolume<std::vector<uint16_t>> )
)

namespace VoxelsLoad
{
MR_CANONICAL_TYPEDEFS( (template <typename T> struct), MRVOXELS_CLASS DicomVolumeT,
    ( DicomVolume, DicomVolumeT<SimpleVolumeMinMax> )
    ( DicomVolumeAsVdb, DicomVolumeT<VdbVolume> )
)
} // namespace VoxelsLoad

} // namespace MR
