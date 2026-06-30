#include "MRWasmBindings.h"

#include "MRVoxels/MRVoxelsLoad.h"
#include "MRVoxels/MRVoxelsVolume.h"

#include <emscripten/bind.h>

#include <filesystem>
#include <string>

using namespace MR;

namespace
{
struct VoxelsLoadModule {};
}

EMSCRIPTEN_BINDINGS( meshlib_voxels_load )
{
    emscripten::class_<VoxelsLoadModule>( "VoxelsLoad" )
        .class_function( "fromAnySupportedFormat", +[]( const std::string& path ) -> emscripten::val
        {
            auto volumes = Wasm::unwrap( VoxelsLoad::fromAnySupportedFormat( std::filesystem::path( path ) ) );
            auto out = emscripten::val::array();
            for ( auto& volume : volumes )
                out.call<void>( "push", volume );
            return out;
        } );
}
