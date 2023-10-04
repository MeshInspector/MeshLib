#pragma once
#ifdef _WIN32
#include "MRMeshFwd.h"

#include "MRExpected.h"

#include <filesystem>
#include <iostream>

namespace MR::MeshLoad
{

MRMESH_API Expected<std::shared_ptr<Object>, std::string> fromSceneStepFile( const std::filesystem::path& path,
                                                                             const ProgressCallback& callback = {} );
MRMESH_API Expected<std::shared_ptr<Object>, std::string> fromSceneStepFile( std::istream& in,
                                                                             const ProgressCallback& callback = {} );

} // namespace MR::MeshLoad
#endif
