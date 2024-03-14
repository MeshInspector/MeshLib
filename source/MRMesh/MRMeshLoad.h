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

/// loads from internal file format
MRMESH_API Expected<Mesh, std::string> fromMrmesh( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );
MRMESH_API Expected<Mesh, std::string> fromMrmesh( std::istream& in, const MeshLoadSettings& settings = {} );

/// loads from .off file
MRMESH_API Expected<Mesh, std::string> fromOff( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );
MRMESH_API Expected<Mesh, std::string> fromOff( std::istream& in, const MeshLoadSettings& settings = {} );

/// loads from .obj file
MRMESH_API Expected<Mesh, std::string> fromObj( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );
/// loads from .obj format
/// important on Windows: in stream must be open in binary mode
MRMESH_API Expected<Mesh, std::string> fromObj( std::istream& in, const MeshLoadSettings& settings = {} );
/// loads from any .stl
MRMESH_API Expected<Mesh, std::string> fromAnyStl( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );
MRMESH_API Expected<Mesh, std::string> fromAnyStl( std::istream& in, const MeshLoadSettings& settings = {} );

/// loads from binary .stl
MRMESH_API Expected<Mesh, std::string> fromBinaryStl( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );
MRMESH_API Expected<Mesh, std::string> fromBinaryStl( std::istream& in, const MeshLoadSettings& settings = {} );

/// loads from ASCII .stl
MRMESH_API Expected<Mesh, std::string> fromASCIIStl( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );
MRMESH_API Expected<Mesh, std::string> fromASCIIStl( std::istream& in, const MeshLoadSettings& settings = {} );

/// loads from .ply file
MRMESH_API Expected<Mesh, std::string> fromPly( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );
MRMESH_API Expected<Mesh, std::string> fromPly( std::istream& in, const MeshLoadSettings& settings = {} );

#ifndef MRMESH_NO_OPENCTM
/// loads from .ctm file
MRMESH_API Expected<Mesh, std::string> fromCtm( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );
MRMESH_API Expected<Mesh, std::string> fromCtm( std::istream& in, const MeshLoadSettings& settings = {} );
#endif

#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_XML )
/// loads from .3mf file (overload that takes path also reads "*.model" files)
MRMESH_API Expected<Mesh, std::string> from3mf( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );
MRMESH_API Expected<Mesh, std::string> from3mf( std::istream& in, const MeshLoadSettings& settings = {} );
#endif

MRMESH_API Expected<Mesh, std::string> fromDxf( const std::filesystem::path& path, const MeshLoadSettings& settings = {} );
MRMESH_API Expected<Mesh, std::string> fromDxf( std::istream& in, const MeshLoadSettings& settings = {} );

/// detects the format from file extension and loads mesh from it
MRMESH_API Expected<Mesh, std::string> fromAnySupportedFormat( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );
/// extension in `*.ext` format
MRMESH_API Expected<Mesh, std::string> fromAnySupportedFormat( std::istream& in, const std::string& extension, const MeshLoadSettings& settings = {} );

/// \}

} // namespace MeshLoad

} // namespace MR
