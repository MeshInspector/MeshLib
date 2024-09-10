#pragma once

#include "config.h"
#ifndef MRIOEXTRAS_NO_STEP
#include "exports.h"

#include <MRMesh/MRExpected.h>
#include <MRMesh/MRMeshLoadSettings.h>

#include <filesystem>
#include <iostream>

namespace MR::MeshLoad
{

/// load mesh data from STEP file using OpenCASCADE
MRIOEXTRAS_API Expected<Mesh> fromStep( const std::filesystem::path& path, const MeshLoadSettings& settings = {} );
MRIOEXTRAS_API Expected<Mesh> fromStep( std::istream& in, const MeshLoadSettings& settings = {} );

/// load scene from STEP file using OpenCASCADE
MRIOEXTRAS_API Expected<std::shared_ptr<Object>> fromSceneStepFile( const std::filesystem::path& path, const MeshLoadSettings& settings = {} );
MRIOEXTRAS_API Expected<std::shared_ptr<Object>> fromSceneStepFile( std::istream& in, const MeshLoadSettings& settings = {} );

} // namespace MR::MeshLoad
#endif
