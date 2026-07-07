#include "MRWasmBindings.h"

#include "MRMesh/MRPointsLoad.h"
#include "MRMesh/MRPointCloud.h"

#include <emscripten/bind.h>

#include <filesystem>
#include <string>

using namespace MR;

namespace
{
struct PointsLoadModule {};
}

EMSCRIPTEN_BINDINGS( meshlib_points_load )
{
    emscripten::class_<PointsLoadModule>( "PointsLoad" )
        .class_function( "fromAnySupportedFormat", +[]( const std::string& path ) -> PointCloud
        {
            return Wasm::unwrap( PointsLoad::fromAnySupportedFormat( std::filesystem::path( path ) ) );
        } );
}
