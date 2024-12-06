#include "MRVDBConversions.h"
#include "MRMesh.h"
#include "detail/TypeCast.h"

#include "MRMesh/MRMesh.h"
#include "MRVoxels/MRVDBConversions.h"

using namespace MR;

REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( FloatGrid )
REGISTER_AUTO_CAST( GridToMeshSettings )
REGISTER_AUTO_CAST2( std::string, MRString )

MRMesh* mrVdbConversionsGridToMesh( const MRFloatGrid* grid_, const MRGridToMeshSettings* settings_, MRString** errorStr )
{
    ARG( grid ); ARG( settings );
    auto res = gridToMesh( grid, settings );
    if ( res )
    {
        RETURN_NEW( *res );
    }

    if ( errorStr && !res )
        *errorStr = auto_cast( new_from( std::move( res.error() ) ) );

    return nullptr;
}
