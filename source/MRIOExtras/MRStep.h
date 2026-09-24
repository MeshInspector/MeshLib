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
    /// assign distinct per-component colors so the scene is visually inspectable;
    /// has no effect if the source STEP file contains any color information
    bool autoColorize = true;
};

/// returns a copy of the STEP load settings applied when a STEP file is loaded through the format
/// registry, i.e. by the generic mesh/scene loading functions that take no STEP-specific arguments;
/// thread-safe: the settings are returned by value under a lock, so a concurrent
/// setStepLoadSettings() call cannot be observed half-applied
MRIOEXTRAS_API StepLoadSettings getStepLoadSettings();

/// sets the STEP load settings applied when a STEP file is loaded through the format registry;
/// does not affect the overloads below that are given \p stepSettings explicitly;
/// thread-safe, but note that concurrent loads through the registry share these settings:
/// each load reads the value that is set when it starts, so change them before starting the load
MRIOEXTRAS_API void setStepLoadSettings( const StepLoadSettings& settings );

/// load mesh data from STEP file using OpenCASCADE
MRIOEXTRAS_API Expected<Mesh> fromStep( const std::filesystem::path& path, const MeshLoadSettings& settings = {}, const StepLoadSettings& stepSettings = {} );
MRIOEXTRAS_API Expected<Mesh> fromStep( std::istream& in, const MeshLoadSettings& settings = {}, const StepLoadSettings& stepSettings = {} );

/// load scene from STEP file using OpenCASCADE
MRIOEXTRAS_API Expected<std::shared_ptr<Object>> fromSceneStepFile( const std::filesystem::path& path, const MeshLoadSettings& settings = {}, const StepLoadSettings& stepSettings = {} );
MRIOEXTRAS_API Expected<std::shared_ptr<Object>> fromSceneStepFile( std::istream& in, const MeshLoadSettings& settings = {}, const StepLoadSettings& stepSettings = {} );

} // namespace MR::MeshLoad
#endif
