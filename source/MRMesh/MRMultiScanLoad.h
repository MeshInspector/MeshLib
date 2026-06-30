#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"

#include <filesystem>

namespace MR::PointsLoad
{

/// \addtogroup PointsLoadGroup
/// \{

/// loads and merges a point cloud from a folder produced by a multi-scan laser capture:
/// the folder is expected to contain pairs of files named _intempNNN.pose and _laserNNN.ply with matching indices NNN
/// (the number of pairs is arbitrary); each .pose file stores a 4x4 row-major rigid transformation,
/// which is applied to the points (and normals) loaded from the .ply file with the same index;
/// \return the union of all transformed points from all found pairs, expressed in the common coordinate frame
MRMESH_API Expected<PointCloud> fromMultiScanFolder( const std::filesystem::path& folder, const ProgressCallback& callback = {} );

/// \}

} // namespace MR::PointsLoad
