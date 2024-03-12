#pragma once

#include "MRMeshFwd.h"
#ifndef MRMESH_NO_OPENCASCADE
#include "MRExpected.h"
#include "MRMeshLoadSettings.h"

#include <filesystem>
#include <iostream>

namespace MR::MeshLoad
{

/// load mesh data from STEP file using OpenCASCADE
MRMESH_API Expected<Mesh> fromStep( const std::filesystem::path& path, const MeshLoadSettings& settings = {} );
MRMESH_API Expected<Mesh> fromStep( std::istream& in, const MeshLoadSettings& settings = {} );

/// load scene from STEP file using OpenCASCADE
MRMESH_API Expected<std::shared_ptr<Object>> fromSceneStepFile( const std::filesystem::path& path, const MeshLoadSettings& settings = {} );
MRMESH_API Expected<std::shared_ptr<Object>> fromSceneStepFile( std::istream& in, const MeshLoadSettings& settings = {} );

} // namespace MR::MeshLoad
#endif
