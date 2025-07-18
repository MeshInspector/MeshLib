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
/// SaveSettings::onlyValidPoints = true is ignored
MRMESH_API Expected<void> toMrmesh( const Mesh & mesh, const std::filesystem::path & file,
                                                     const SaveSettings & settings = {} );
MRMESH_API Expected<void> toMrmesh( const Mesh & mesh, std::ostream & out,
                                                     const SaveSettings & settings = {} );

/// saves in .off file
MRMESH_API Expected<void> toOff( const Mesh & mesh, const std::filesystem::path & file,
                                                  const SaveSettings & settings = {} );
MRMESH_API Expected<void> toOff( const Mesh & mesh, std::ostream & out,
                                                  const SaveSettings & settings = {} );

/// saves in .obj file
/// \param firstVertId is the index of first mesh vertex in the output file (if this object is not the first there)
MRMESH_API Expected<void> toObj( const Mesh & mesh, const std::filesystem::path & file,
                                                  const SaveSettings & settings, int firstVertId );
MRMESH_API Expected<void> toObj( const Mesh & mesh, std::ostream & out,
                                                  const SaveSettings & settings, int firstVertId );
MRMESH_API Expected<void> toObj( const Mesh & mesh, const std::filesystem::path & file, const SaveSettings & settings = {} );
MRMESH_API Expected<void> toObj( const Mesh & mesh, std::ostream & out, const SaveSettings & settings = {} );

/// saves in binary .stl file;
/// SaveSettings::onlyValidPoints = false is ignored
MRMESH_API Expected<void> toBinaryStl( const Mesh & mesh, const std::filesystem::path & file, const SaveSettings & settings = {} );
MRMESH_API Expected<void> toBinaryStl( const Mesh & mesh, std::ostream & out, const SaveSettings & settings = {} );

/// saves in textual .stl file;
/// SaveSettings::onlyValidPoints = false is ignored
MRMESH_API Expected<void> toAsciiStl( const Mesh& mesh, const std::filesystem::path& file, const SaveSettings & settings = {} );
MRMESH_API Expected<void> toAsciiStl( const Mesh& mesh, std::ostream& out, const SaveSettings & settings = {} );

/// saves in .ply file
MRMESH_API Expected<void> toPly( const Mesh& mesh, const std::filesystem::path& file, const SaveSettings & settings = {} );
MRMESH_API Expected<void> toPly( const Mesh & mesh, std::ostream & out, const SaveSettings & settings = {} );

/// detects the format from file extension and save mesh to it
MRMESH_API Expected<void> toAnySupportedFormat( const Mesh & mesh, const std::filesystem::path & file, const SaveSettings & settings = {} );
/// extension in `*.ext` format
MRMESH_API Expected<void> toAnySupportedFormat( const Mesh& mesh, const std::string& extension, std::ostream& out, const SaveSettings & settings = {} );

/// \}

} // namespace MeshSave

} // namespace MR
