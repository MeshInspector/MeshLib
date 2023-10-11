#pragma once

#include "MRMeshFwd.h"
#include "MRIOFilters.h"
#include "MRId.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"
#include "MRMeshLoadSettings.h"
#include <filesystem>
#include <istream>
#include <string>

namespace MR
{

namespace MeshLoad
{

/// \defgroup MeshLoadGroup Mesh Load
/// \ingroup IOGroup
/// \{

#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_XML )
/// loads from .model 3MF file
MRMESH_API Expected<Mesh, std::string> from3mfModel( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );
MRMESH_API Expected<Mesh, std::string> from3mfModel( std::istream& in, const MeshLoadSettings& settings = {} );
/// loads from .3mf file
MRMESH_API Expected<Mesh, std::string> from3mf( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );
MRMESH_API Expected<Mesh, std::string> from3mf( std::istream& in, const MeshLoadSettings& settings = {} );
#endif

/// \}

} // namespace MeshLoad

} // namespace MR
