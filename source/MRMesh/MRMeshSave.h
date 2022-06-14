#pragma once

#include "MRMeshFwd.h"
#include <tl/expected.hpp>
#include <filesystem>
#include <ostream>
#include <string>
#include "MRIOFilters.h"
#include "MRAffineXf3.h"

namespace MR
{

namespace MeshSave
{

/// \defgroup MeshSaveGroup Mesh Save
/// \ingroup IOGroup
/// \{

MRMESH_API extern const IOFilters Filters;

/// saves in internal file format
MRMESH_API tl::expected<void, std::string> toMrmesh( const Mesh & mesh, const std::filesystem::path & file );
MRMESH_API tl::expected<void, std::string> toMrmesh( const Mesh & mesh, std::ostream & out );

/// saves in .off file
MRMESH_API tl::expected<void, std::string> toOff( const Mesh & mesh, const std::filesystem::path & file );
MRMESH_API tl::expected<void, std::string> toOff( const Mesh & mesh, std::ostream & out );

/// saves in .obj file
/// \param xf will be applied to all mesh vertices prior to saving;
/// \param firstVertId is the index of first mesh vertex in the output file (if this object is not the first there)
MRMESH_API tl::expected<void, std::string> toObj( const Mesh & mesh, const std::filesystem::path & file, const AffineXf3f & xf = {}, int firstVertId = 1 );
MRMESH_API tl::expected<void, std::string> toObj( const Mesh & mesh, std::ostream & out, const AffineXf3f & xf = {}, int firstVertId = 1 );

/// saves in binary .stl file
MRMESH_API tl::expected<void, std::string> toBinaryStl( const Mesh & mesh, const std::filesystem::path & file );
MRMESH_API tl::expected<void, std::string> toBinaryStl( const Mesh & mesh, std::ostream & out );

/// saves in .ply file
MRMESH_API tl::expected<void, std::string> toPly( const Mesh& mesh, const std::filesystem::path& file, const Vector<Color, VertId>* colors = nullptr );
MRMESH_API tl::expected<void, std::string> toPly( const Mesh & mesh, std::ostream & out, const Vector<Color, VertId>* colors = nullptr );

struct CtmSaveOptions
{
    enum class MeshCompression
    {
        None,     ///< no compression at all, fast but not effective
        Lossless, ///< compression without any loss in vertex coordinates
        Lossy     ///< compression with loss in vertex coordinates
    };
    MeshCompression meshCompression = MeshCompression::Lossless;
    /// fixed point precision for vertex coordinates in case of MeshCompression::Lossy. 
    /// For instance, if this value is 0.001, all vertex coordinates will be rounded to three decimals
    float vertexPrecision = 1.0f / 1024.0f; //~= 0.00098
    /// LZMA compression: 0 - minimal compression, but fast; 9 - maximal compression, but slow
    int compressionLevel = 1; 
    /// if it is turned on, then higher compression ratios are reached but the order of triangles is changed
    bool rearrangeTriangles = false;
    /// comment saved in the file
    const char * comment = "MeshInspector.com";
};

/// saves in .ctm file
MRMESH_API tl::expected<void, std::string> toCtm( const Mesh & mesh, const std::filesystem::path & file, const CtmSaveOptions options = {}, const Vector<Color, VertId>* colors = nullptr );
MRMESH_API tl::expected<void, std::string> toCtm( const Mesh & mesh, std::ostream & out, const CtmSaveOptions options = {}, const Vector<Color, VertId>* colors = nullptr );

/// detects the format from file extension and save mesh to it
MRMESH_API tl::expected<void, std::string> toAnySupportedFormat( const Mesh & mesh, const std::filesystem::path & file, const Vector<Color, VertId>* colors = nullptr );
/// extension in `*.ext` format
MRMESH_API tl::expected<void, std::string> toAnySupportedFormat( const Mesh& mesh, std::ostream& out, const std::string& extension, const Vector<Color, VertId>* colors = nullptr );

/// \}

} // namespace MeshSave

} // namespace MR
