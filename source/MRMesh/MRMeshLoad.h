#pragma once

#include "MRMeshFwd.h"
#include "MRIOFilters.h"
#include "MRId.h"
#include <tl/expected.hpp>
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
MRMESH_API tl::expected<Mesh, std::string> fromMrmesh( const std::filesystem::path& file, Vector<Color, VertId>* colors = nullptr );
MRMESH_API tl::expected<Mesh, std::string> fromMrmesh( std::istream& in, Vector<Color, VertId>* colors = nullptr );

/// loads from .off file
MRMESH_API tl::expected<Mesh, std::string> fromOff( const std::filesystem::path& file, Vector<Color, VertId>* colors = nullptr );
MRMESH_API tl::expected<Mesh, std::string> fromOff( std::istream& in, Vector<Color, VertId>* colors = nullptr );

/// loads from .obj file
MRMESH_API tl::expected<Mesh, std::string> fromObj( const std::filesystem::path& file, Vector<Color, VertId>* colors = nullptr );
MRMESH_API tl::expected<Mesh, std::string> fromObj( std::istream& in, Vector<Color, VertId>* colors = nullptr );
/// loads from any .stl
MRMESH_API tl::expected<Mesh, std::string> fromAnyStl( const std::filesystem::path& file, Vector<Color, VertId>* colors = nullptr );
MRMESH_API tl::expected<Mesh, std::string> fromAnyStl( std::istream& in, Vector<Color, VertId>* colors = nullptr );

/// loads from binary .stl
MRMESH_API tl::expected<Mesh, std::string> fromBinaryStl( const std::filesystem::path & file, Vector<Color, VertId>* colors = nullptr );
MRMESH_API tl::expected<Mesh, std::string> fromBinaryStl( std::istream & in, Vector<Color, VertId>* colors = nullptr );

/// loads from ASCII .stl
MRMESH_API tl::expected<Mesh, std::string> fromASCIIStl( const std::filesystem::path& file, Vector<Color, VertId>* colors = nullptr );
MRMESH_API tl::expected<Mesh, std::string> fromASCIIStl( std::istream& in, Vector<Color, VertId>* colors = nullptr );

/// loads from .ply file
MRMESH_API tl::expected<Mesh, std::string> fromPly( const std::filesystem::path& file, Vector<Color, VertId>* colors = nullptr );
MRMESH_API tl::expected<Mesh, std::string> fromPly( std::istream& in, Vector<Color, VertId>* colors = nullptr );

/// loads from .ctm file
MRMESH_API tl::expected<Mesh, std::string> fromCtm( const std::filesystem::path & file, Vector<Color, VertId>* colors = nullptr );
MRMESH_API tl::expected<Mesh, std::string> fromCtm( std::istream & in, Vector<Color, VertId>* colors = nullptr );

/// detects the format from file extension and loads mesh from it
MRMESH_API tl::expected<Mesh, std::string> fromAnySupportedFormat( const std::filesystem::path& file, Vector<Color, VertId>* colors = nullptr );
/// extension in `*.ext` format
MRMESH_API tl::expected<Mesh, std::string> fromAnySupportedFormat( std::istream& in, const std::string& extension, Vector<Color, VertId>* colors = nullptr );

/// \}

} // namespace MeshLoad

} // namespace MR
