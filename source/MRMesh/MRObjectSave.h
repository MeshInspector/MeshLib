#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"

#include <filesystem>

namespace MR::ObjectSave
{

/// save a scene object to a given file
/// if the file format is scene-capable, saves all the object's entities
/// otherwise, saves only merged entities of the corresponding type (mesh, polyline, point cloud, etc.)
MRMESH_API Expected<void> toAnySupportedFormat( const Object& object, const std::filesystem::path& file,
                                                ProgressCallback callback = {} );

} // namespace MR::ObjectSave
