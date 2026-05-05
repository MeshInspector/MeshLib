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

/// STEP-specific mesh load parameters
struct StepLoadSettings
{
    /// angular deflection used to tessellate the boundary edges
    double angularDeflection = 0.1;
    /// linear deflection used to tessellate the boundary edges
    double linearDeflection = 0.5;
    /// whether the deflection values are related to the size of edges
    bool relative = false;
};

/// load mesh data from STEP file using OpenCASCADE
MRIOEXTRAS_API Expected<Mesh> fromStep( const std::filesystem::path& path, const MeshLoadSettings& settings = {}, const StepLoadSettings& stepSettings = {} );
MRIOEXTRAS_API Expected<Mesh> fromStep( std::istream& in, const MeshLoadSettings& settings = {}, const StepLoadSettings& stepSettings = {} );

/// load scene from STEP file using OpenCASCADE
MRIOEXTRAS_API Expected<std::shared_ptr<Object>> fromSceneStepFile( const std::filesystem::path& path, const MeshLoadSettings& settings = {}, const StepLoadSettings& stepSettings = {} );
MRIOEXTRAS_API Expected<std::shared_ptr<Object>> fromSceneStepFile( std::istream& in, const MeshLoadSettings& settings = {}, const StepLoadSettings& stepSettings = {} );

} // namespace MR::MeshLoad
#endif
