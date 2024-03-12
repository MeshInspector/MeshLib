#pragma once
#include "MRMeshFwd.h"
#ifndef __EMSCRIPTEN__
#include "MRExpected.h"
#include "MRMeshLoadSettings.h"
#include <filesystem>

namespace MR
{

/// load scene from MISON file
MRMESH_API Expected<std::shared_ptr<Object>, std::string> fromSceneMison( const std::filesystem::path& path, 
    std::string* loadWarn = nullptr, ProgressCallback callback = {} );
MRMESH_API Expected<std::shared_ptr<Object>, std::string> fromSceneMison( std::istream& in, 
    std::string* loadWarn = nullptr, ProgressCallback callback = {} );

}
#endif