#pragma once

#include "MRMeshFwd.h"
#include <tl/expected.hpp>
#include <filesystem>
#include <ostream>
#include <string>
#include "MRIOFilters.h"

namespace MR
{

namespace MeshSave
{

MRMESH_API extern const IOFilters Filters;

// saves in internal file format
MRMESH_API tl::expected<void, std::string> toMrmesh( const Mesh & mesh, const std::filesystem::path & file );
MRMESH_API tl::expected<void, std::string> toMrmesh( const Mesh & mesh, std::ostream & out );

// saves in .off file
MRMESH_API tl::expected<void, std::string> toOff( const Mesh & mesh, const std::filesystem::path & file );
MRMESH_API tl::expected<void, std::string> toOff( const Mesh & mesh, std::ostream & out );

// saves in .obj file
MRMESH_API tl::expected<void, std::string> toObj( const Mesh & mesh, const std::filesystem::path & file );
MRMESH_API tl::expected<void, std::string> toObj( const Mesh & mesh, std::ostream & out );

// saves in binary .stl file
MRMESH_API tl::expected<void, std::string> toBinaryStl( const Mesh & mesh, const std::filesystem::path & file );
MRMESH_API tl::expected<void, std::string> toBinaryStl( const Mesh & mesh, std::ostream & out );

// saves in .ply file
MRMESH_API tl::expected<void, std::string> toPly( const Mesh& mesh, const std::filesystem::path& file, const std::vector<Color>* perVertColors = nullptr );
MRMESH_API tl::expected<void, std::string> toPly( const Mesh & mesh, std::ostream & out, const std::vector<Color>* perVertColors = nullptr );

struct CtmSaveOptions
{
    enum class MeshCompression
    {
        None,     // no compression at all, fast but not effective
        Lossless, // compression without any loss in vertex coordinates
        Lossy     // compression with loss in vertex coordinates
    };
    MeshCompression meshCompression = MeshCompression::Lossless;
    // fixed point precision for vertex coordinates in case of MeshCompression::Lossy. 
    // For instance, if this value is 0.001, all vertex coordinates will be rounded to three decimals
    float vertexPrecision = 1.0f / 1024.0f; //~= 0.00098
    // LZMA compression: 0 - minimal compression, but fast; 9 - maximal compression, but slow
    int compressionLevel = 1; 
    // if it is turned on, then higher compression ratios are reached but the order of triangles is changed
    bool rearrangeTriangles = false;
    // comment saved in the file
    const char * comment = "MeshRUs";
};

// saves in .ctm file
MRMESH_API tl::expected<void, std::string> toCtm( const Mesh & mesh, const std::filesystem::path & file, const CtmSaveOptions options = {} );
MRMESH_API tl::expected<void, std::string> toCtm( const Mesh & mesh, std::ostream & out, const CtmSaveOptions options = {} );

// detects the format from file extension and save mesh to it
MRMESH_API tl::expected<void, std::string> toAnySupportedFormat( const Mesh & mesh, const std::filesystem::path & file, const std::vector<Color>* perVertColors = nullptr );
// extension in `*.ext` format
MRMESH_API tl::expected<void, std::string> toAnySupportedFormat( const Mesh& mesh, std::ostream& out, const std::string& extension, const std::vector<Color>* perVertColors = nullptr );

} //namespace MeshSave

} //namespace MR
