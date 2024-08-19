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
MRMESH_API Expected<Mesh> fromMrmesh( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );
MRMESH_API Expected<Mesh> fromMrmesh( std::istream& in, const MeshLoadSettings& settings = {} );

/// loads from .off file
MRMESH_API Expected<Mesh> fromOff( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );
MRMESH_API Expected<Mesh> fromOff( std::istream& in, const MeshLoadSettings& settings = {} );

/// loads from .obj file
MRMESH_API Expected<Mesh> fromObj( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );
/// loads from .obj format
/// important on Windows: in stream must be open in binary mode
MRMESH_API Expected<Mesh> fromObj( std::istream& in, const MeshLoadSettings& settings = {} );
/// loads from any .stl
MRMESH_API Expected<Mesh> fromAnyStl( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );
MRMESH_API Expected<Mesh> fromAnyStl( std::istream& in, const MeshLoadSettings& settings = {} );

/// loads from binary .stl
MRMESH_API Expected<Mesh> fromBinaryStl( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );
MRMESH_API Expected<Mesh> fromBinaryStl( std::istream& in, const MeshLoadSettings& settings = {} );

/// loads from ASCII .stl
MRMESH_API Expected<Mesh> fromASCIIStl( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );
MRMESH_API Expected<Mesh> fromASCIIStl( std::istream& in, const MeshLoadSettings& settings = {} );

/// loads from .ply file
MRMESH_API Expected<Mesh> fromPly( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );
MRMESH_API Expected<Mesh> fromPly( std::istream& in, const MeshLoadSettings& settings = {} );

#ifndef MRMESH_NO_OPENCTM
/// loads from .ctm file
MRMESH_API Expected<Mesh> fromCtm( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );
MRMESH_API Expected<Mesh> fromCtm( std::istream& in, const MeshLoadSettings& settings = {} );
#endif

MRMESH_API Expected<Mesh> fromDxf( const std::filesystem::path& path, const MeshLoadSettings& settings = {} );
MRMESH_API Expected<Mesh> fromDxf( std::istream& in, const MeshLoadSettings& settings = {} );

/// detects the format from file extension and loads mesh from it
MRMESH_API Expected<Mesh> fromAnySupportedFormat( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );
/// extension in `*.ext` format
MRMESH_API Expected<Mesh> fromAnySupportedFormat( std::istream& in, const std::string& extension, const MeshLoadSettings& settings = {} );

/// \}

} // namespace MeshLoad

} // namespace MR
