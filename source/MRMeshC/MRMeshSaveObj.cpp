#include "MRMeshSaveObj.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRMeshSaveObj.h"

using namespace MR;

namespace
{

// pass the deleter to non-owning shared pointers
struct null_deleter
{
    void operator ()( void const * ) { }
};

} // namespace

REGISTER_AUTO_CAST( AffineXf3f )
REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST2( std::string, MRString )

void mrMeshSaveSceneToObj( const MRMeshSaveNamedXfMesh* objects_, size_t objectsNum, const char* file, MRString** errorString )
{
    std::vector<MeshSave::NamedXfMesh> objects;
    objects.reserve( objectsNum );
    for ( auto i = 0; i < objectsNum; ++i )
    {
        auto& object_ = objects_[i];
        objects.push_back( {
            .name = object_.name,
            .toWorld = auto_cast( object_.toWorld ),
            .mesh = std::shared_ptr<const Mesh>( auto_cast( object_.mesh ), null_deleter{} ),
        } );
    }

    auto result = MeshSave::sceneToObj( objects, file );
    if ( !result.has_value() && errorString )
    {
        *errorString = auto_cast( new_from( std::move( result.error() ) ) );
    }
}
