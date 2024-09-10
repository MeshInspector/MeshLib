#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRIOFilters.h"
#include "MRSaveSettings.h"
#include <filesystem>
#include <ostream>

namespace MR
{

namespace MeshSave
{

/// \defgroup MeshSaveGroup Mesh Save
/// \ingroup IOGroup
/// \{

/// saves in internal file format;
/// SaveSettings::saveValidOnly = true is ignored
MRMESH_API VoidOrErrStr toMrmesh( const Mesh & mesh, const std::filesystem::path & file,
                                                     const SaveSettings & settings = {} );
MRMESH_API VoidOrErrStr toMrmesh( const Mesh & mesh, std::ostream & out,
                                                     const SaveSettings & settings = {} );

/// saves in .off file
MRMESH_API VoidOrErrStr toOff( const Mesh & mesh, const std::filesystem::path & file,
                                                  const SaveSettings & settings = {} );
MRMESH_API VoidOrErrStr toOff( const Mesh & mesh, std::ostream & out,
                                                  const SaveSettings & settings = {} );

/// saves in .obj file
/// \param firstVertId is the index of first mesh vertex in the output file (if this object is not the first there)
MRMESH_API VoidOrErrStr toObj( const Mesh & mesh, const std::filesystem::path & file,
                                                  const SaveSettings & settings, int firstVertId );
MRMESH_API VoidOrErrStr toObj( const Mesh & mesh, std::ostream & out,
                                                  const SaveSettings & settings, int firstVertId );
MRMESH_API VoidOrErrStr toObj( const Mesh & mesh, const std::filesystem::path & file, const SaveSettings & settings = {} );
MRMESH_API VoidOrErrStr toObj( const Mesh & mesh, std::ostream & out, const SaveSettings & settings = {} );

/// saves in binary .stl file;
/// SaveSettings::saveValidOnly = false is ignored
MRMESH_API VoidOrErrStr toBinaryStl( const Mesh & mesh, const std::filesystem::path & file, const SaveSettings & settings = {} );
MRMESH_API VoidOrErrStr toBinaryStl( const Mesh & mesh, std::ostream & out, const SaveSettings & settings = {} );

/// saves in textual .stl file;
/// SaveSettings::saveValidOnly = false is ignored
MRMESH_API VoidOrErrStr toAsciiStl( const Mesh& mesh, const std::filesystem::path& file, const SaveSettings & settings = {} );
MRMESH_API VoidOrErrStr toAsciiStl( const Mesh& mesh, std::ostream& out, const SaveSettings & settings = {} );

/// saves in .ply file
MRMESH_API VoidOrErrStr toPly( const Mesh& mesh, const std::filesystem::path& file, const SaveSettings & settings = {} );
MRMESH_API VoidOrErrStr toPly( const Mesh & mesh, std::ostream & out, const SaveSettings & settings = {} );

/// detects the format from file extension and save mesh to it
MRMESH_API VoidOrErrStr toAnySupportedFormat( const Mesh & mesh, const std::filesystem::path & file, const SaveSettings & settings = {} );
/// extension in `*.ext` format
MRMESH_API VoidOrErrStr toAnySupportedFormat( const Mesh& mesh, const std::string& extension, std::ostream& out, const SaveSettings & settings = {} );

/// \}

} // namespace MeshSave

} // namespace MR
