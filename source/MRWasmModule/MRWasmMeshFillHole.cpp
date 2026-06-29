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
        .property( "maxPolygonSubdivisions", &FillHoleParams::maxPolygonSubdivisions );

    emscripten::function( "fillHole", +[]( std::shared_ptr<Mesh> m, int edge, const FillHoleParams& params )
    {
        fillHole( *m, EdgeId( edge ), params );
    } );

    emscripten::function( "fillHoles", +[]( std::shared_ptr<Mesh> m, emscripten::val edges, const FillHoleParams& params )
    {
        const size_t len = edges[ "length" ].as<size_t>();
        std::vector<EdgeId> es( len );
        for ( size_t i = 0; i < len; ++i )
            es[i] = EdgeId( edges[ i ].as<int>() );
        fillHoles( *m, es, params );
    } );
}
