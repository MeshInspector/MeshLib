#include "MRWasmBindings.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshSubdivide.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_subdivide )
{
    emscripten::class_<SubdivideSettings>( "SubdivideSettings" )
        .constructor<>()
        .property( "maxEdgeLen", &SubdivideSettings::maxEdgeLen )
        .property( "curvaturePriority", &SubdivideSettings::curvaturePriority )
        .property( "maxEdgeSplits", &SubdivideSettings::maxEdgeSplits )
        .property( "maxDeviationAfterFlip", &SubdivideSettings::maxDeviationAfterFlip )
        .property( "maxAngleChangeAfterFlip", &SubdivideSettings::maxAngleChangeAfterFlip )
        .property( "criticalAspectRatioFlip", &SubdivideSettings::criticalAspectRatioFlip )
        .property( "subdivideBorder", &SubdivideSettings::subdivideBorder )
        .property( "maxTriAspectRatio", &SubdivideSettings::maxTriAspectRatio )
        .property( "maxSplittableTriAspectRatio", &SubdivideSettings::maxSplittableTriAspectRatio )
        .property( "smoothMode", &SubdivideSettings::smoothMode )
        .property( "minSharpDihedralAngle", &SubdivideSettings::minSharpDihedralAngle )
        .property( "projectOnOriginalMesh", &SubdivideSettings::projectOnOriginalMesh );

    emscripten::function( "subdivideMesh", +[]( std::shared_ptr<Mesh> mesh, const SubdivideSettings& settings )
    {
        return subdivideMesh( *mesh, settings );
    } );
}
