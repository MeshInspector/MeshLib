#include "MRPointsLoad.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRPointsLoad.h"

using namespace MR;

REGISTER_AUTO_CAST( PointCloud )
REGISTER_AUTO_CAST2( std::string, MRString )

MRPointCloud* mrPointsLoadFromAnySupportedFormat( const char* filename, MRString** errorString )
{
    auto res = PointsLoad::fromAnySupportedFormat( filename );
    if ( res )
    {
        RETURN_NEW( std::move( *res ) );
    }
    else
    {
        if ( errorString )
            *errorString = auto_cast( new_from( std::move( res.error() ) ) );
        return NULL;
    }
}
