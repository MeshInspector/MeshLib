#include "MRMeshSaveObj.h"
#include "MRMeshSave.h"
#include "MRStringConvert.h"

namespace MR
{

namespace MeshSave
{

VoidOrErrStr sceneToObj( const std::vector<NamedXfMesh> & objects, const std::filesystem::path & file )
{
    std::ofstream out( file );
    if ( !out )
        return tl::make_unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return sceneToObj( objects, out );
}

VoidOrErrStr sceneToObj( const std::vector<NamedXfMesh> & objects, std::ostream & out )
{
    out << "# MeshInspector.com\n";
    int firstVertId = 1;
    for ( auto & o : objects )
    {
        assert( o.mesh );
        if ( !o.mesh )
            continue;
        out << "o " << o.name << '\n';
        auto saveRes = toObj( *o.mesh, out, o.toWorld, firstVertId );
        if ( !saveRes.has_value() )
            return saveRes; //error
        firstVertId += o.mesh->topology.lastValidVert() + 1;
    }
    return {}; //success
}

} // namespace MeshSave

} // namespace MR
