#include "MRMeshLoad.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshLoad.h"

using namespace MR;

REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST2( std::string, MRString )

MRMesh* mrMeshLoadFromAnySupportedFormat( const char* file, MRString** errorStr )
{
    auto res = MeshLoad::fromAnySupportedFormat( file );
    if ( res )
    {
        RETURN_NEW( std::move( *res ) );
    }
    if ( errorStr )
    {
        *errorStr = auto_cast( new_from( std::move( res.error() ) ) );
    }
    return nullptr;
}
