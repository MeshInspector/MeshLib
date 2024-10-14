#include "MRMeshSave.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshSave.h"

using namespace MR;

REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST2( std::string, MRString )

void mrMeshSaveToAnySupportedFormat( const MRMesh* mesh_, const char* file, MRString** errorStr )
{
    ARG( mesh );
    auto res = MeshSave::toAnySupportedFormat( mesh, file );
    if ( !res && errorStr )
    {
        *errorStr = auto_cast( new_from( std::move( res.error() ) ) );
    }
}
