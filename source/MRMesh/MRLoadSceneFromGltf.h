#pragma once
#include "MRMeshFwd.h"
#include "MRObject.h"

namespace MR
{
//loads scene from glTF file in a new container object
MRMESH_API tl::expected<std::shared_ptr<Object>, std::string> deserializeObjectTreeFromGltf( const std::filesystem::path& file, ProgressCallback callback = {} );
}
