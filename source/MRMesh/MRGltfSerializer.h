#pragma once

#ifndef MRMESH_NO_GLTF
#include "MRMeshFwd.h"
#include "MRObject.h"

namespace MR
{
//loads scene from glTF file in a new container object
MRMESH_API tl::expected<std::shared_ptr<Object>, std::string> deserializeObjectTreeFromGltf( const std::filesystem::path& file, ProgressCallback callback = {} );
//saves scene to a glTF file
MRMESH_API tl::expected<void, std::string> serializeObjectTreeToGltf( const Object& root, const std::filesystem::path& file, ProgressCallback callback = {} );

}
#endif
