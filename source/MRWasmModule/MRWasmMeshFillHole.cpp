#include "MRWasmBindings.h"

#include "MRMesh/MRMeshFillHole.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRId.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <memory>
#include <vector>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_mesh_fill_hole )
{
    emscripten::enum_<FillHoleParams::MultipleEdgesResolveMode>( "MultipleEdgesResolveMode" )
        .value( "None", FillHoleParams::MultipleEdgesResolveMode::None )
        .value( "Simple", FillHoleParams::MultipleEdgesResolveMode::Simple )
        .value( "Strong", FillHoleParams::MultipleEdgesResolveMode::Strong );

    emscripten::class_<FillHoleParams>( "FillHoleParams" )
        .constructor<>()
        .property( "smoothBd", &FillHoleParams::smoothBd )
        .property( "multipleEdgesResolveMode", &FillHoleParams::multipleEdgesResolveMode )
        .property( "makeDegenerateBand", &FillHoleParams::makeDegenerateBand )
        .property( "maxPolygonSubdivisions", &FillHoleParams::maxPolygonSubdivisions )
        .property( "metric", &FillHoleParams::metric );

    emscripten::class_<StitchHolesParams>( "StitchHolesParams" )
        .constructor<>()
        .property( "metric", &StitchHolesParams::metric );

    emscripten::function( "fillHole", +[]( std::shared_ptr<Mesh> mesh, int a, const FillHoleParams& params )
    {
        fillHole( *mesh, EdgeId( a ), params );
    } );

    emscripten::function( "fillHoles", +[]( std::shared_ptr<Mesh> mesh, emscripten::val as, const FillHoleParams& params )
    {
        const size_t len = as[ "length" ].as<size_t>();
        std::vector<EdgeId> es( len );
        for ( size_t i = 0; i < len; ++i )
            es[i] = EdgeId( as[ i ].as<int>() );
        fillHoles( *mesh, es, params );
    } );

    emscripten::function( "stitchHoles", +[]( std::shared_ptr<Mesh> mesh, int a, int b, const StitchHolesParams& params )
    {
        stitchHoles( *mesh, EdgeId( a ), EdgeId( b ), params );
    } );
}
