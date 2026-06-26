#include "MRWasmBindings.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRFixSelfIntersections.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_fix_self_intersections )
{
    emscripten::enum_<SelfIntersections::Settings::Method>( "SelfIntersectionsMethod" )
        .value( "Relax", SelfIntersections::Settings::Method::Relax )
        .value( "CutAndFill", SelfIntersections::Settings::Method::CutAndFill );

    emscripten::class_<SelfIntersections::Settings>( "SelfIntersectionsSettings" )
        .constructor<>()
        .property( "touchIsIntersection", &SelfIntersections::Settings::touchIsIntersection )
        .property( "method", &SelfIntersections::Settings::method )
        .property( "relaxIterations", &SelfIntersections::Settings::relaxIterations )
        .property( "maxExpand", &SelfIntersections::Settings::maxExpand )
        .property( "subdivideEdgeLen", &SelfIntersections::Settings::subdivideEdgeLen )
        .property( "mimicPatch", &SelfIntersections::Settings::mimicPatch )
        .property( "callback",
            +[]( const SelfIntersections::Settings& ) { return emscripten::val::undefined(); },
            +[]( SelfIntersections::Settings& s, emscripten::val cb ) { s.callback = Wasm::jsToCppCallback( cb ); } );

    emscripten::function( "fixSelfIntersections", +[]( Mesh& mesh, const SelfIntersections::Settings& settings )
    {
        Wasm::unwrap( SelfIntersections::fix( mesh, settings ) );
    } );
}
