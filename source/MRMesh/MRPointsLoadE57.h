#pragma once

#include "MRPointCloud.h"
#include "MRExpected.h"
#include "MRProgressCallback.h"
#include "MRAffineXf3.h"
#include <filesystem>
#include <string>

namespace MR
{

namespace PointsLoad
{

/// loads scene from e57 file
struct NamedCloud
{
    std::string name;
    PointCloud cloud;
    AffineXf3f xf;
    VertColors colors;
};
MRMESH_API Expected<std::vector<NamedCloud>> fromSceneE57File( const std::filesystem::path& file, bool combineAllObjects, ProgressCallback progress = {} );

} // namespace PointsLoad

} // namespace MR
