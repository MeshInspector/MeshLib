#include "MRWasmBindings.h"

#include "MRMesh/MRNormalDenoising.h"
#include "MRMesh/MRMesh.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_normal_denoising )
{
    emscripten::class_<DenoiseViaNormalsSettings>( "DenoiseViaNormalsSettings" )
        .constructor<>()
        .property( "fastIndicatorComputation", &DenoiseViaNormalsSettings::fastIndicatorComputation )
        .property( "beta", &DenoiseViaNormalsSettings::beta )
        .property( "gamma", &DenoiseViaNormalsSettings::gamma )
        .property( "normalIters", &DenoiseViaNormalsSettings::normalIters )
        .property( "pointIters", &DenoiseViaNormalsSettings::pointIters )
        .property( "guideWeight", &DenoiseViaNormalsSettings::guideWeight )
        .property( "limitNearInitial", &DenoiseViaNormalsSettings::limitNearInitial )
        .property( "maxInitialDist", &DenoiseViaNormalsSettings::maxInitialDist );

    emscripten::function( "meshDenoiseViaNormals", +[]( std::shared_ptr<Mesh> mesh )
    {
        Wasm::unwrap( meshDenoiseViaNormals( *mesh ) );
    } );
    emscripten::function( "meshDenoiseViaNormals", +[]( std::shared_ptr<Mesh> mesh, const DenoiseViaNormalsSettings& settings )
    {
        Wasm::unwrap( meshDenoiseViaNormals( *mesh, settings ) );
    } );
}
