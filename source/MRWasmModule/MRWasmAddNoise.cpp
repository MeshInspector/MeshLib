#include "MRWasmBindings.h"

#include "MRMesh/MRAddNoise.h"
#include "MRMesh/MRMesh.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_add_noise )
{
    emscripten::class_<NoiseSettings>( "NoiseSettings" )
        .constructor<>()
        .property( "sigma", &NoiseSettings::sigma )
        .property( "seed", &NoiseSettings::seed );

    emscripten::function( "addNoise", +[]( std::shared_ptr<Mesh> mesh, const NoiseSettings& settings )
    {
        Wasm::unwrap( addNoise( *mesh, nullptr, settings ) );
    } );
}
