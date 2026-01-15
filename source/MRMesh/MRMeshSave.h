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

/// saver in binary .stl format that can write triangles one by one not knowing their number beforehand
class BinaryStlSaver
{
public:
    /// writes STL header in the stream
    MRMESH_API explicit BinaryStlSaver( std::ostream & out, const SaveSettings & settings = {}, std::uint32_t expectedNumTris = 0 );

    /// writes one more triangle in the stream
    MRMESH_API bool writeTri( const Triangle3f& tri );

    /// if initially written the number of triangles do not match to the actual number of written triangles,
    /// updates the number in the header
    MRMESH_API bool updateHeadCounter();

    /// calls updateHeadCounter();
    MRMESH_API ~BinaryStlSaver();

private:
    std::ostream & out_;
    const SaveSettings & settings_;
    std::ostream::pos_type numTrisPos_ = 0; ///< the location in the stream where the number of triangles is stored
    std::uint32_t headNumTris_ = 0;  ///< what was written in the header
    std::uint32_t savedNumTris_ = 0; ///< how many triangles were actually written in the stream
};

/// saves in textual .stl file;
/// SaveSettings::onlyValidPoints = false is ignored
MRMESH_API Expected<void> toAsciiStl( const Mesh & mesh, const std::filesystem::path& file, const SaveSettings & settings = {} );
MRMESH_API Expected<void> toAsciiStl( const Mesh & mesh, std::ostream& out, const SaveSettings & settings = {} );

/// saves in .ply file
MRMESH_API Expected<void> toPly( const Mesh & mesh, const std::filesystem::path& file, const SaveSettings & settings = {} );
MRMESH_API Expected<void> toPly( const Mesh & mesh, std::ostream & out, const SaveSettings & settings = {} );

/// saves in 3mf .model file
MRMESH_API Expected<void> toModel3mf( const Mesh & mesh, const std::filesystem::path& file, const SaveSettings & settings = {} );
MRMESH_API Expected<void> toModel3mf( const Mesh & mesh, std::ostream & out, const SaveSettings & settings = {} );

/// saves in .3mf file
MRMESH_API Expected<void> to3mf( const Mesh & mesh, const std::filesystem::path& file, const SaveSettings & settings = {} );

/// detects the format from file extension and save mesh to it
MRMESH_API Expected<void> toAnySupportedFormat( const Mesh & mesh, const std::filesystem::path & file, const SaveSettings & settings = {} );
/// extension in `*.ext` format
MRMESH_API Expected<void> toAnySupportedFormat( const Mesh & mesh, const std::string& extension, std::ostream& out, const SaveSettings & settings = {} );

/// \}

} // namespace MeshSave

} // namespace MR
