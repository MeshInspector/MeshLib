#pragma once

#include "config.h"
#ifndef MRIOEXTRAS_NO_3MF
#include "exports.h"

#include <MRMesh/MRExpected.h>
#include <MRMesh/MRObject.h>

namespace MR
{

// loads scene from 3MF file in a new container object
MRIOEXTRAS_API Expected<std::shared_ptr<Object>> deserializeObjectTreeFrom3mf( const std::filesystem::path& file, std::string* loadWarn = nullptr, ProgressCallback callback = {} );

// loads scene from .model file in a new container object
MRIOEXTRAS_API Expected<std::shared_ptr<Object>> deserializeObjectTreeFromModel( const std::filesystem::path& file, std::string* loadWarn = nullptr, ProgressCallback callback = {} );

} // namespace MR
#endif
