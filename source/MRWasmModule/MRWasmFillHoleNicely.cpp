#include "MRWasmBindings.h"

#include "MRMesh/MRFillHoleNicely.h"
#include "MRMesh/MRMeshFillHole.h"
#include "MRMesh/MREnums.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRBitSet.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_fill_hole_nicely )
{
    emscripten::class_<SubdivideFillingSettings>( "SubdivideFillingSettings" )
        .constructor<>()
        .property( "maxEdgeLen", &SubdivideFillingSettings::maxEdgeLen )
        .property( "maxEdgeSplits", &SubdivideFillingSettings::maxEdgeSplits )
        .property( "maxAngleChangeAfterFlip", &SubdivideFillingSettings::maxAngleChangeAfterFlip );

    emscripten::class_<SmoothFillingSettings>( "SmoothFillingSettings" )
        .constructor<>()
        .property( "naturalSmooth", &SmoothFillingSettings::naturalSmooth )
        .property( "edgeWeights", &SmoothFillingSettings::edgeWeights )
        .property( "vmass", &SmoothFillingSettings::vmass );

    emscripten::class_<FillHoleNicelySettings>( "FillHoleNicelySettings" )
        .constructor<>()
        .property( "triangulateParams", &FillHoleNicelySettings::triangulateParams )
        .property( "triangulateOnly", &FillHoleNicelySettings::triangulateOnly )
        .property( "subdivideSettings", &FillHoleNicelySettings::subdivideSettings )
        .property( "smoothCurvature", &FillHoleNicelySettings::smoothCurvature )
        .property( "smoothSettings", &FillHoleNicelySettings::smoothSettings );

    emscripten::function( "fillHoleNicely", +[]( std::shared_ptr<Mesh> mesh, int holeEdge, const FillHoleNicelySettings& settings )
    {
        return fillHoleNicely( *mesh, EdgeId( holeEdge ), settings );
    } );
}
