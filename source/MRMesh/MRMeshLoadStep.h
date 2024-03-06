#pragma once

#include "MRMeshFwd.h"
#ifndef MRMESH_NO_OPENCASCADE
#include "MRExpected.h"
#include "MRMeshLoadSettings.h"

#include <filesystem>
#include <iostream>

namespace MR::MeshLoad
{

/// load scene from STEP file using OpenCASCADE
MRMESH_API Expected<std::shared_ptr<Object>, std::string> fromSceneStepFile( const std::filesystem::path& path, const MeshLoadSettings& settings = {} );
MRMESH_API Expected<std::shared_ptr<Object>, std::string> fromSceneStepFile( std::istream& in, const MeshLoadSettings& settings = {} );

/// ...
MRMESH_API Expected<std::shared_ptr<Object>> fromSceneStepFile2( const std::filesystem::path& path, const MeshLoadSettings& settings = {} );

} // namespace MR::MeshLoad
#endif
