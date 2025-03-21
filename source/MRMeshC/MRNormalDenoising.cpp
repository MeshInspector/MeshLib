#include "MRNormalDenoising.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRNormalDenoising.h"

using namespace MR;

REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( UndirectedEdgeBitSet )
REGISTER_AUTO_CAST2( std::string, MRString )

#define COPY_FROM( obj, field ) . field = auto_cast( ( obj ). field ),

MRDenoiseViaNormalsSettings mrDenoiseViaNormalsSettingsNew()
{
    static const DenoiseViaNormalsSettings def;
    return {
        COPY_FROM( def, fastIndicatorComputation )
        COPY_FROM( def, beta )
        COPY_FROM( def, gamma )
        COPY_FROM( def, normalIters )
        COPY_FROM( def, pointIters )
        COPY_FROM( def, guideWeight )
        COPY_FROM( def, limitNearInitial )
        COPY_FROM( def, maxInitialDist )
        COPY_FROM( def, outCreases )
        .cb = NULL,
    };
}

void mrMeshDenoiseViaNormals( MRMesh* mesh_, const MRDenoiseViaNormalsSettings* settings_, MRString** errorString )
{
    ARG( mesh );

    DenoiseViaNormalsSettings settings;
    if ( settings_ )
    {
        settings = {
            COPY_FROM( *settings_, fastIndicatorComputation )
            COPY_FROM( *settings_, beta )
            COPY_FROM( *settings_, gamma )
            COPY_FROM( *settings_, normalIters )
            COPY_FROM( *settings_, pointIters )
            COPY_FROM( *settings_, guideWeight )
            COPY_FROM( *settings_, limitNearInitial )
            COPY_FROM( *settings_, maxInitialDist )
            COPY_FROM( *settings_, outCreases )
            .cb = settings_->cb,
        };
    }

    auto result = meshDenoiseViaNormals( mesh, settings );
    if ( errorString && !result )
        *errorString = auto_cast( new_from( std::move( result.error() ) ) );
}
