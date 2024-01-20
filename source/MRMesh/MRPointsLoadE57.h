#pragma once

#include "MRPointCloud.h"
#include "MRExpected.h"
#include "MRProgressCallback.h"
#include "MRAffineXf3.h"
#include <filesystem>
#include <string>

#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_E57 )

namespace MR
{

namespace PointsLoad
{

struct E57LoadSettings
{
    /// true => if input file has more than one cloud, they all will be combined in one
    bool combineAllObjects = false;

    /// true => return only identity transforms, applying them to points
    bool identityXf = false;

    /// progress report and cancellation
    ProgressCallback progress;
};

/// loads scene from e57 file
struct NamedCloud
{
    std::string name;
    PointCloud cloud;
    AffineXf3f xf;
    VertColors colors;
};

MRMESH_API Expected<std::vector<NamedCloud>> fromSceneE57File( const std::filesystem::path& file,
    const E57LoadSettings & settings = {} );

} // namespace PointsLoad

} // namespace MR

#endif //!defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_E57 )
