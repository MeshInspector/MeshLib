#pragma once
#include "MRMeshFwd.h"
#ifndef __EMSCRIPTEN__
#include "MRExpected.h"
#include "MRMeshLoadSettings.h"
#include "MRLoadedObjects.h"
#include <filesystem>

namespace MR
{

/// load scene from MISON file \n
/// JSON file with array named "Objects" or root array: \n
/// element fields:\n
///    "Filename" : required full path to file for loading object
///    "XF": optional xf for loaded object
///    "Name": optional name for loaded object
MRMESH_API Expected<LoadedObject> fromSceneMison( const std::filesystem::path& path, const ProgressCallback& callback = {} );
MRMESH_API Expected<LoadedObject> fromSceneMison( std::istream& in, const ProgressCallback& callback = {} );

} //namespace MR

#endif //!__EMSCRIPTEN__
