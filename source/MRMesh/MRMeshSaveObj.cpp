#include "MRMeshSaveObj.h"
#include "MRMeshSave.h"
#include "MRStringConvert.h"

namespace MR
{

namespace MeshSave
{

tl::expected<void, std::string> sceneToObj( const std::vector<NamedXfMesh> & objects, const std::filesystem::path & file )
{
    std::ofstream out( file );
    if ( !out )
        return tl::make_unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return sceneToObj( objects, out );
}

tl::expected<void, std::string> sceneToObj( const std::vector<NamedXfMesh> & objects, std::ostream & out )
{
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
