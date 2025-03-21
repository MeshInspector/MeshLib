#include "MRUniformSampling.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRUniformSampling.h"
#include "MRMesh/MRBitSet.h"

using namespace MR;

REGISTER_AUTO_CAST( PointCloud )
REGISTER_AUTO_CAST( VertBitSet )

#define COPY_FROM( obj, field ) . field = auto_cast( ( obj ). field ),

MRUniformSamplingSettings mrUniformSamplingSettingsNew()
{
    static const UniformSamplingSettings def;
    return {
        COPY_FROM( def, distance )
        COPY_FROM( def, minNormalDot )
        COPY_FROM( def, lexicographicalOrder )
        .progress = NULL,
    };
}

MRVertBitSet* mrPointUniformSampling( const MRPointCloud* pointCloud_, const MRUniformSamplingSettings* settings_ )
{
    ARG( pointCloud );

    UniformSamplingSettings settings;
    if ( settings_ )
    {
        settings = {
            COPY_FROM( *settings_, distance )
            COPY_FROM( *settings_, minNormalDot )
            COPY_FROM( *settings_, lexicographicalOrder )
            .progress = settings_->progress,
        };
    }

    auto result = pointUniformSampling( pointCloud, settings );
    if ( result )
        RETURN_NEW( std::move( *result ) );
    else
        return NULL;
}
