#pragma once

#include "MRMeshFwd.h"

#ifndef MRMESH_NO_XML
#include "MRObject.h"
#include "MRExpected.h"

namespace MR
{
//loads scene from 3MF file in a new container object
MRMESH_API Expected<std::shared_ptr<Object>> deserializeObjectTreeFrom3mf( const std::filesystem::path& file, std::string* loadWarn = nullptr, ProgressCallback callback = {} );

//loads scene from .model file in a new container object
MRMESH_API Expected<std::shared_ptr<Object>> deserializeObjectTreeFromModel( const std::filesystem::path& file, std::string* loadWarn = nullptr, ProgressCallback callback = {} );

}
#endif