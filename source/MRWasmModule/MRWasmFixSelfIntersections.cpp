#include "MRWasmBindings.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRFixSelfIntersections.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

using namespace emscripten;

namespace
{

void fixSelfIntersections( MR::Mesh& mesh, const MR::SelfIntersections::Settings& settings )
{
    unwrap( MR::SelfIntersections::fix( mesh, settings ) );
}

}

EMSCRIPTEN_BINDINGS( meshlib_fix_self_intersections )
{
    enum_<MR::SelfIntersections::Settings::Method>( "SelfIntersectionsMethod" )
        .value( "Relax", MR::SelfIntersections::Settings::Method::Relax )
        .value( "CutAndFill", MR::SelfIntersections::Settings::Method::CutAndFill );

    class_<MR::SelfIntersections::Settings>( "SelfIntersectionsSettings" )
        .constructor<>()
        .property( "touchIsIntersection", &MR::SelfIntersections::Settings::touchIsIntersection )
        .property( "method", &MR::SelfIntersections::Settings::method )
        .property( "relaxIterations", &MR::SelfIntersections::Settings::relaxIterations )
        .property( "maxExpand", &MR::SelfIntersections::Settings::maxExpand )
        .property( "subdivideEdgeLen", &MR::SelfIntersections::Settings::subdivideEdgeLen )
        .property( "mimicPatch", &MR::SelfIntersections::Settings::mimicPatch )
        .property( "callback",
            +[]( const MR::SelfIntersections::Settings& ) { return val::undefined(); },
            +[]( MR::SelfIntersections::Settings& s, val cb ) { s.callback = jsToCppCallback( cb ); } );

    function( "fixSelfIntersections", &fixSelfIntersections );
}
