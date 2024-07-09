#include "MRPointsLoad.h"

#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRPointsLoad.h"

using namespace MR;

MRPointCloud* mrPointsLoadFromAnySupportedFormat( const char* filename, MRString** errorString )
{
    auto res = PointsLoad::fromAnySupportedFormat( filename );
    if ( res )
    {
        return reinterpret_cast<MRPointCloud*>( new PointCloud( std::move( *res ) ) );
    }
    else
    {
        if ( errorString )
            *errorString = reinterpret_cast<MRString*>( new std::string( std::move( res.error() ) ) );
        return NULL;
    }
}
