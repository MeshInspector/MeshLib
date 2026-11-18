#include "MRWasmBindings.h"

#include "MRVoxels/MRVoxelsSave.h"
#include "MRVoxels/MRVoxelsVolume.h"

#include <emscripten/bind.h>

#include <filesystem>
#include <string>

using namespace MR;

namespace
{
struct VoxelsSaveModule {};
}

EMSCRIPTEN_BINDINGS( meshlib_voxels_save )
{
    emscripten::class_<VoxelsSaveModule>( "VoxelsSave" )
        .class_function( "toAnySupportedFormat", +[]( const VdbVolume& volume, const std::string& path )
        {
            Wasm::unwrap( VoxelsSave::toAnySupportedFormat( volume, std::filesystem::path( path ) ) );
        } );
}
