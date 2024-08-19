#pragma once
#include "MRMeshFwd.h"
#ifndef __EMSCRIPTEN__
#include "MRExpected.h"
#include "MRMeshLoadSettings.h"
#include <filesystem>

namespace MR
{

/// load scene from MISON file \n
/// JSON file with array named "Objects" or root array: \n
/// element fields:\n
///    "Filename" : required full path to file for loading object
///    "XF": optional xf for loaded object
///    "Name": optional name for loaded object
MRMESH_API Expected<std::shared_ptr<Object>> fromSceneMison( const std::filesystem::path& path, 
    std::string* loadWarn = nullptr, ProgressCallback callback = {} );
MRMESH_API Expected<std::shared_ptr<Object>> fromSceneMison( std::istream& in, 
    std::string* loadWarn = nullptr, ProgressCallback callback = {} );

}
#endif