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

EMSCRIPTEN_DECLARE_VAL_TYPE( VdbVolumeArrayVal )

EMSCRIPTEN_BINDINGS( meshlib_voxels_load )
{
    emscripten::register_type<VdbVolumeArrayVal>( "VdbVolume[]" );

    emscripten::class_<VoxelsLoadModule>( "VoxelsLoad" )
        .class_function( "fromAnySupportedFormat", +[]( const std::string& path ) -> VdbVolumeArrayVal
        {
            auto volumes = Wasm::unwrap( VoxelsLoad::fromAnySupportedFormat( std::filesystem::path( path ) ) );
            auto out = emscripten::val::array();
            for ( auto& volume : volumes )
                out.call<void>( "push", volume );
            return VdbVolumeArrayVal( out );
        } );
}
