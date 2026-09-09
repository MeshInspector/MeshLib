#include "MRWasmBindings.h"

#include "MRMesh/MRUniformSampling.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRBitSet.h"

#include <emscripten/bind.h>

#include <optional>
#include <utility>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_uniform_sampling )
{
    emscripten::class_<UniformSamplingSettings>( "UniformSamplingSettings" )
        .constructor<>()
        .property( "distance", &UniformSamplingSettings::distance )
        .property( "minNormalDot", &UniformSamplingSettings::minNormalDot )
        .property( "lexicographicalOrder", &UniformSamplingSettings::lexicographicalOrder );

    emscripten::function( "pointUniformSampling", +[]( const PointCloud& pointCloud, const UniformSamplingSettings& settings )
    {
        auto res = pointUniformSampling( pointCloud, settings );
        if ( !res )
            Wasm::throwJsError( "pointUniformSampling was cancelled" );
        return std::move( *res );
    } );
}
