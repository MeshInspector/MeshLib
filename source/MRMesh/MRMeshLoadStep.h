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
/// load just a bunch of meshes
MRMESH_API Expected<std::shared_ptr<Object>, std::string> fromSceneStepFile( const std::filesystem::path& path, const MeshLoadSettings& settings = {} );
MRMESH_API Expected<std::shared_ptr<Object>, std::string> fromSceneStepFile( std::istream& in, const MeshLoadSettings& settings = {} );

/// load scene from STEP file using OpenCASCADE
/// preserve internal objects' structure and names
MRMESH_API Expected<std::shared_ptr<Object>> fromSceneStepFileEx( const std::filesystem::path& path, const MeshLoadSettings& settings = {} );
MRMESH_API Expected<std::shared_ptr<Object>> fromSceneStepFileEx( std::istream& in, const MeshLoadSettings& settings = {} );

} // namespace MR::MeshLoad
#endif
