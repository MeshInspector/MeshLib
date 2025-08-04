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

class FloatGrid;

MR_CANONICAL_TYPEDEFS( (template <typename T> struct), MRVOXELS_CLASS VoxelsVolumeMinMax,
    ( SimpleVolumeMinMax, VoxelsVolumeMinMax<Vector<float, VoxelId>> )
    ( SimpleVolumeMinMaxU16, VoxelsVolumeMinMax<Vector<uint16_t, VoxelId>> )
    ( VdbVolume, VoxelsVolumeMinMax<FloatGrid> )
)

using VdbVolumes = std::vector<VdbVolume>;

template <typename T>
using VoxelValueGetter = std::function<T ( const Vector3i& )>;

MR_CANONICAL_TYPEDEFS( (template <typename T> struct), MRVOXELS_CLASS VoxelsVolume,
    ( FunctionVolume, VoxelsVolume<VoxelValueGetter<float>> )
    ( FunctionVolumeU8, VoxelsVolume<VoxelValueGetter<uint8_t>> )
    ( SimpleVolume, VoxelsVolume<Vector<float, VoxelId>> )
    ( SimpleVolumeU16, VoxelsVolume<Vector<uint16_t, VoxelId>> )
    ( SimpleBinaryVolume, VoxelsVolume<VoxelBitSet> )
)

namespace VoxelsLoad
{
MR_CANONICAL_TYPEDEFS( (template <typename T> struct), MRVOXELS_CLASS DicomVolumeT,
    ( DicomVolume, DicomVolumeT<SimpleVolumeMinMax> )
    ( DicomVolumeAsVdb, DicomVolumeT<VdbVolume> )
)
} // namespace VoxelsLoad

} // namespace MR
