#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"

#include <filesystem>
#include <string>

namespace MR::PointsLoad
{

/// \addtogroup PointsLoadGroup
/// \{

/// parameters of \ref fromMultiScanFolder
struct MultiScanLoadSettings
{
    /// file name prefix (before the numeric index) of .pose files, each storing a 4x4 row-major transformation
    std::string posePrefix = "pose";
    /// file name prefix (before the numeric index) of .ply files, each storing the points of one scan
    std::string scanPrefix = "scan";
    /// to report progress and to cancel the operation
    ProgressCallback callback;
};

/// loads and merges a point cloud from a folder with the results of a multi-scan capture:
/// the folder is expected to contain pairs of files named <posePrefix>NNN.pose and <scanPrefix>NNN.ply
/// with matching indices NNN (the number of pairs is arbitrary); each .pose file stores a 4x4 row-major
/// rigid transformation, which is applied to the points (and normals) loaded from the .ply file with the same index;
/// \return the union of all transformed points from all found pairs, expressed in the common coordinate frame
MRMESH_API Expected<PointCloud> fromMultiScanFolder( const std::filesystem::path& folder, const MultiScanLoadSettings& settings = {} );

/// \}

} // namespace MR::PointsLoad
