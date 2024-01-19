#pragma once

#include "MRPointCloud.h"
#include "MRExpected.h"
#include "MRProgressCallback.h"
#include <filesystem>
#include <string>

namespace MR
{

namespace PointsLoad
{

/// \defgroup MeshLoadObjGroup Mesh Load Obj
/// \ingroup IOGroup
/// \{

/// loads scene from e57 file
struct NamedCloud
{
    std::string name;
    PointCloud cloud;
    VertColors colors;
};
MRMESH_API Expected<std::vector<NamedCloud>> fromSceneE57File( const std::filesystem::path& file, bool combineAllObjects,
                                                               AffineXf3f* outXf = nullptr, ProgressCallback progress = {} );

} // namespace PointsLoad

} // namespace MR
