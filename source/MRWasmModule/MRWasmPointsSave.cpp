#include "MRWasmBindings.h"

#include "MRMesh/MRPointsSave.h"
#include "MRMesh/MRPointCloud.h"

#include <emscripten/bind.h>

#include <filesystem>
#include <string>

using namespace MR;

namespace
{
struct PointsSaveModule {};
}

EMSCRIPTEN_BINDINGS( meshlib_points_save )
{
    emscripten::class_<PointsSaveModule>( "PointsSave" )
        .class_function( "toAnySupportedFormat", +[]( const PointCloud& points, const std::string& path )
        {
            Wasm::unwrap( PointsSave::toAnySupportedFormat( points, std::filesystem::path( path ) ) );
        } );
}
