#pragma once

#include "MRMeshFwd.h"
#ifndef MRMESH_NO_GLTF
#include "MRObject.h"
#include "MRExpected.h"

namespace MR
{
//loads scene from glTF file in a new container object
MRMESH_API Expected<std::shared_ptr<Object>, std::string> deserializeObjectTreeFromGltf( const std::filesystem::path& file, ProgressCallback callback = {} );
//saves scene to a glTF file
MRMESH_API VoidOrErrStr serializeObjectTreeToGltf( const Object& root, const std::filesystem::path& file, ProgressCallback callback = {} );

}
#endif
