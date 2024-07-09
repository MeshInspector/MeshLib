#include "MRPointsSave.h"

#include "MRMesh/MRPointsSave.h"

using namespace MR;

void mrPointsSaveToAnySupportedFormat( const MRPointCloud* pc_, const char* file, MRString** errorString )
{
    const auto& pc = *reinterpret_cast<const PointCloud*>( pc_ );

    auto res = PointsSave::toAnySupportedFormat( pc, file );
    if ( !res && errorString != nullptr )
        *errorString = reinterpret_cast<MRString*>( new std::string( std::move( res.error() ) ) );
}
