#pragma once

#include "config.h"
#ifndef MRIOEXTRAS_NO_GLTF
#include "exports.h"

#include <MRMesh/MRExpected.h>
#include <MRMesh/MRObject.h>

namespace MR
{

// loads scene from glTF file in a new container object
MRIOEXTRAS_API Expected<std::shared_ptr<Object>> deserializeObjectTreeFromGltf( const std::filesystem::path& file, ProgressCallback callback = {} );
// saves scene to a glTF file
MRIOEXTRAS_API Expected<void> serializeObjectTreeToGltf( const Object& root, const std::filesystem::path& file, ProgressCallback callback = {} );

} // namespace MR
#endif
