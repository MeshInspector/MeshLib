#include "MRWasmBindings.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshDecimate.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_decimate )
{
    emscripten::enum_<DecimateStrategy>( "DecimateStrategy" )
        .value( "MinimizeError", DecimateStrategy::MinimizeError )
        .value( "ShortestEdgeFirst", DecimateStrategy::ShortestEdgeFirst );

    emscripten::value_object<DecimateResult>( "DecimateResult" )
        .field( "vertsDeleted", &DecimateResult::vertsDeleted )
        .field( "facesDeleted", &DecimateResult::facesDeleted )
        .field( "errorIntroduced", &DecimateResult::errorIntroduced )
        .field( "cancelled", &DecimateResult::cancelled );

    emscripten::class_<DecimateSettings>( "DecimateSettings" )
        .constructor<>()
        .property( "strategy", &DecimateSettings::strategy )
        .property( "maxError", &DecimateSettings::maxError )
        .property( "maxEdgeLen", &DecimateSettings::maxEdgeLen )
        .property( "maxBdShift", &DecimateSettings::maxBdShift )
        .property( "maxTriangleAspectRatio", &DecimateSettings::maxTriangleAspectRatio )
        .property( "criticalTriAspectRatio", &DecimateSettings::criticalTriAspectRatio )
        .property( "tinyEdgeLength", &DecimateSettings::tinyEdgeLength )
        .property( "stabilizer", &DecimateSettings::stabilizer )
        .property( "angleWeightedDistToPlane", &DecimateSettings::angleWeightedDistToPlane )
        .property( "optimizeVertexPos", &DecimateSettings::optimizeVertexPos )
        .property( "maxDeletedVertices", &DecimateSettings::maxDeletedVertices )
        .property( "maxDeletedFaces", &DecimateSettings::maxDeletedFaces )
        .property( "collapseNearNotFlippable", &DecimateSettings::collapseNearNotFlippable )
        .property( "touchNearBdEdges", &DecimateSettings::touchNearBdEdges )
        .property( "touchBdVerts", &DecimateSettings::touchBdVerts )
        .property( "maxAngleChange", &DecimateSettings::maxAngleChange )
        .property( "packMesh", &DecimateSettings::packMesh )
        .property( "subdivideParts", &DecimateSettings::subdivideParts )
        .property( "decimateBetweenParts", &DecimateSettings::decimateBetweenParts )
        .property( "minFacesInPart", &DecimateSettings::minFacesInPart );

    emscripten::function( "decimateMesh", +[]( std::shared_ptr<Mesh> mesh, const DecimateSettings& settings )
    {
        return decimateMesh( *mesh, settings );
    } );

    emscripten::class_<RemeshSettings>( "RemeshSettings" )
        .constructor<>()
        .property( "targetEdgeLen", &RemeshSettings::targetEdgeLen )
        .property( "maxEdgeSplits", &RemeshSettings::maxEdgeSplits )
        .property( "maxAngleChangeAfterFlip", &RemeshSettings::maxAngleChangeAfterFlip )
        .property( "frozenBoundary", &RemeshSettings::frozenBoundary )
        .property( "maxBdShift", &RemeshSettings::maxBdShift )
        .property( "useCurvature", &RemeshSettings::useCurvature )
        .property( "maxSplittableTriAspectRatio", &RemeshSettings::maxSplittableTriAspectRatio )
        .property( "finalRelaxIters", &RemeshSettings::finalRelaxIters )
        .property( "finalRelaxNoShrinkage", &RemeshSettings::finalRelaxNoShrinkage )
        .property( "packMesh", &RemeshSettings::packMesh )
        .property( "projectOnOriginalMesh", &RemeshSettings::projectOnOriginalMesh );

    emscripten::function( "remesh", +[]( std::shared_ptr<Mesh> m, const RemeshSettings& s )
    {
        return remesh( *m, s );
    } );
}
