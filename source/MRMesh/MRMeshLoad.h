#pragma once

#include "MRMeshFwd.h"
#include "MRIOFilters.h"
#include "MRId.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"
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
MRMESH_API Expected<Mesh, std::string> fromMrmesh( const std::filesystem::path& file, VertColors* colors = nullptr,
                                                       ProgressCallback callback = {} );
MRMESH_API Expected<Mesh, std::string> fromMrmesh( std::istream& in, VertColors* colors = nullptr,
                                                       ProgressCallback callback = {} );

/// loads from .off file
MRMESH_API Expected<Mesh, std::string> fromOff( const std::filesystem::path& file, VertColors* colors = nullptr,
                                                    ProgressCallback callback = {} );
MRMESH_API Expected<Mesh, std::string> fromOff( std::istream& in, VertColors* colors = nullptr,
                                                    ProgressCallback callback = {} );

/// loads from .obj file
MRMESH_API Expected<Mesh, std::string> fromObj( const std::filesystem::path& file, VertColors* colors = nullptr,
                                                    ProgressCallback callback = {} );
/// loads from .obj format
/// important on Windows: in stream must be open in binary mode
MRMESH_API Expected<Mesh, std::string> fromObj( std::istream& in, VertColors* colors = nullptr,
                                                    ProgressCallback callback = {} );
/// loads from any .stl
MRMESH_API Expected<Mesh, std::string> fromAnyStl( const std::filesystem::path& file, VertColors* colors = nullptr,
                                                       ProgressCallback callback = {} );
MRMESH_API Expected<Mesh, std::string> fromAnyStl( std::istream& in, VertColors* colors = nullptr,
                                                       ProgressCallback callback = {} );

/// loads from binary .stl
MRMESH_API Expected<Mesh, std::string> fromBinaryStl( const std::filesystem::path& file, VertColors* colors = nullptr,
                                                          ProgressCallback callback = {} );
MRMESH_API Expected<Mesh, std::string> fromBinaryStl( std::istream& in, VertColors* colors = nullptr,
                                                          ProgressCallback callback = {} );

/// loads from ASCII .stl
MRMESH_API Expected<Mesh, std::string> fromASCIIStl( const std::filesystem::path& file, VertColors* colors = nullptr,
                                                         ProgressCallback callback = {} );
MRMESH_API Expected<Mesh, std::string> fromASCIIStl( std::istream& in, VertColors* colors = nullptr,
                                                         ProgressCallback callback = {} );

/// loads from .ply file
MRMESH_API Expected<Mesh, std::string> fromPly( const std::filesystem::path& file, VertColors* colors = nullptr,
                                                    ProgressCallback callback = {} );
MRMESH_API Expected<Mesh, std::string> fromPly( std::istream& in, VertColors* colors = nullptr,
                                                    ProgressCallback callback = {} );

#ifndef MRMESH_NO_OPENCTM
/// loads from .ctm file
MRMESH_API Expected<Mesh, std::string> fromCtm( const std::filesystem::path& file, VertColors* colors = nullptr,
                                                    ProgressCallback callback = {} );
MRMESH_API Expected<Mesh, std::string> fromCtm( std::istream& in, VertColors* colors = nullptr,
                                                    ProgressCallback callback = {} );
#endif

#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_XML )
/// loads from .model 3MF file
MRMESH_API Expected<Mesh, std::string> from3mfModel( const std::filesystem::path& file, VertColors* colors = nullptr,
                                                    ProgressCallback callback = {} );
MRMESH_API Expected<Mesh, std::string> from3mfModel( std::istream& in, VertColors* colors = nullptr,
                                                    ProgressCallback callback = {} );
#endif

/// detects the format from file extension and loads mesh from it
MRMESH_API Expected<Mesh, std::string> fromAnySupportedFormat( const std::filesystem::path& file, VertColors* colors = nullptr,
                                                                   ProgressCallback callback = {} );
/// extension in `*.ext` format
MRMESH_API Expected<Mesh, std::string> fromAnySupportedFormat( std::istream& in, const std::string& extension, VertColors* colors = nullptr,
                                                                   ProgressCallback callback = {} );

/// \}

} // namespace MeshLoad

} // namespace MR
