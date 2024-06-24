#pragma once

#include "MRMeshFwd.h"

#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_XML )
#include "MRObject.h"
#include "MRExpected.h"

namespace MR
{
//loads scene from glTF file in a new container object
MRMESH_API Expected<std::shared_ptr<Object>, std::string> deserializeObjectTreeFrom3mf( const std::filesystem::path& file, std::string* loadWarn = nullptr, ProgressCallback callback = {} );

}
#endif