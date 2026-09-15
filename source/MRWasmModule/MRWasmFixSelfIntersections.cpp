#include "MRWasmBindings.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRFixSelfIntersections.h"
#include "MRMesh/MRBitSet.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

using namespace MR;

namespace
{
struct SelfIntersectionsModule {};
}

EMSCRIPTEN_DECLARE_VAL_TYPE( SelfIntersectionsCallbackVal )

EMSCRIPTEN_BINDINGS( meshlib_fix_self_intersections )
{
    emscripten::register_type<SelfIntersectionsCallbackVal>( "( progress: number ) => boolean" );

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
            +[]( const SelfIntersections::Settings& ) -> SelfIntersectionsCallbackVal { return SelfIntersectionsCallbackVal( emscripten::val::undefined() ); },
            +[]( SelfIntersections::Settings& s, SelfIntersectionsCallbackVal cb ) { s.callback = Wasm::jsToCppCallback( cb ); } );

    emscripten::class_<SelfIntersectionsModule>( "SelfIntersections" )
        .class_function( "getFaces", +[]( const Mesh& mesh, bool touchIsIntersection )
        {
            return Wasm::unwrap( SelfIntersections::getFaces( mesh, touchIsIntersection ) );
        } )
        .class_function( "fix", +[]( Mesh& mesh, const SelfIntersections::Settings& settings )
        {
            Wasm::unwrap( SelfIntersections::fix( mesh, settings ) );
        } );
}
