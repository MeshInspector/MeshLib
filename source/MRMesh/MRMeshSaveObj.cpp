#include "MRMeshSaveObj.h"
#include "MRMeshSave.h"
#include "MRStringConvert.h"
#include "MRColor.h"

namespace MR
{

namespace MeshSave
{

Expected<void> sceneToObj( const std::vector<NamedXfMesh> & objects, const std::filesystem::path & file,
                         VertColors* colors )
{
    // although .obj is a textual format, we open the file in binary mode to get exactly the same result on Windows and Linux
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return sceneToObj( objects, out, colors );
}

Expected<void> sceneToObj( const std::vector<NamedXfMesh> & objects, std::ostream & out, VertColors* colors )
{
    out << "# MeshInspector.com\n";
    int firstVertId = 1;
    for ( auto & o : objects )
    {
        assert( o.mesh );
        if ( !o.mesh )
            continue;
        out << "o " << o.name << '\n';
        AffineXf3d xf( o.toWorld );
        auto saveRes = toObj( *o.mesh, out, { .colors = colors, .xf = &xf }, firstVertId );
        if ( !saveRes.has_value() )
            return saveRes; //error
        firstVertId += o.mesh->topology.numValidVerts();
    }
    return {}; //success
}

} // namespace MeshSave

} // namespace MR
