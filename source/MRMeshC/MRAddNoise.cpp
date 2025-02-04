#include "MRAddNoise.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRAddNoise.h"

using namespace MR;

REGISTER_AUTO_CAST( Mesh );
REGISTER_AUTO_CAST( VertBitSet );
REGISTER_AUTO_CAST2( std::string, MRString )

#define COPY_FROM( obj, field ) . field = ( obj ). field ,

MRNoiseSettings mrNoiseSettingsNew()
{
    static const NoiseSettings def;
    return {
        COPY_FROM( def, sigma )
        COPY_FROM( def, seed )
    };
}

void mrAddNoiseToMesh( MRMesh *mesh_, const MRVertBitSet *region_, const MRNoiseSettings *settings_, MRString **errorString )
{
    ARG( mesh ); ARG_PTR( region );

    NoiseSettings settings;
    if ( settings_ )
    {
        settings = {
            COPY_FROM( *settings_, sigma )
            COPY_FROM( *settings_, seed )
            COPY_FROM( *settings_, callback )
        };
    }

    auto result = addNoise( mesh, region, settings );
    if ( errorString && !result )
        *errorString = auto_cast( new_from( std::move( result.error() ) ) );
}
