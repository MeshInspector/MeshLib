#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"

#include <filesystem>

namespace MR::ObjectSave
{

/// save an object tree to a given file
/// file format must be scene-capable
MRMESH_API Expected<void> toAnySupportedSceneFormat( const Object& object, const std::filesystem::path& file,
                                                     ProgressCallback callback = {} );

/// save a scene object to a given file
/// if the file format is scene-capable, saves all the object's entities
/// otherwise, saves only merged entities of the corresponding type (mesh, polyline, point cloud, etc.)
MRMESH_API Expected<void> toAnySupportedFormat( const Object& object, const std::filesystem::path& file,
                                                ProgressCallback callback = {} );

} // namespace MR::ObjectSave
