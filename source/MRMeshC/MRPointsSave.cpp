#include "MRPointsSave.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRPointsSave.h"

using namespace MR;

REGISTER_AUTO_CAST( PointCloud )
REGISTER_AUTO_CAST2( std::string, MRString )

void mrPointsSaveToAnySupportedFormat( const MRPointCloud* pc_, const char* file, MRString** errorString )
{
    ARG( pc );
    auto res = PointsSave::toAnySupportedFormat( pc, file );
    if ( !res && errorString != nullptr )
        *errorString = auto_cast( new_from( std::move( res.error() ) ) );
}
