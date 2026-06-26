#include "MRWasmBindings.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshDecimate.h"

#include <emscripten/bind.h>

#include <memory>

using namespace emscripten;

namespace
{

MR::DecimateResult decimateMeshWrap( std::shared_ptr<MR::Mesh> mesh, const MR::DecimateSettings& settings )
{
    return MR::decimateMesh( *mesh, settings );
}

}

EMSCRIPTEN_BINDINGS( meshlib_decimate )
{
    enum_<MR::DecimateStrategy>( "DecimateStrategy" )
        .value( "MinimizeError", MR::DecimateStrategy::MinimizeError )
        .value( "ShortestEdgeFirst", MR::DecimateStrategy::ShortestEdgeFirst );

    value_object<MR::DecimateResult>( "DecimateResult" )
        .field( "vertsDeleted", &MR::DecimateResult::vertsDeleted )
        .field( "facesDeleted", &MR::DecimateResult::facesDeleted )
        .field( "errorIntroduced", &MR::DecimateResult::errorIntroduced )
        .field( "cancelled", &MR::DecimateResult::cancelled );

    class_<MR::DecimateSettings>( "DecimateSettings" )
        .constructor<>()
        .property( "strategy", &MR::DecimateSettings::strategy )
        .property( "maxError", &MR::DecimateSettings::maxError )
        .property( "maxEdgeLen", &MR::DecimateSettings::maxEdgeLen )
        .property( "maxBdShift", &MR::DecimateSettings::maxBdShift )
        .property( "maxTriangleAspectRatio", &MR::DecimateSettings::maxTriangleAspectRatio )
        .property( "criticalTriAspectRatio", &MR::DecimateSettings::criticalTriAspectRatio )
        .property( "tinyEdgeLength", &MR::DecimateSettings::tinyEdgeLength )
        .property( "stabilizer", &MR::DecimateSettings::stabilizer )
        .property( "angleWeightedDistToPlane", &MR::DecimateSettings::angleWeightedDistToPlane )
        .property( "optimizeVertexPos", &MR::DecimateSettings::optimizeVertexPos )
        .property( "maxDeletedVertices", &MR::DecimateSettings::maxDeletedVertices )
        .property( "maxDeletedFaces", &MR::DecimateSettings::maxDeletedFaces )
        .property( "collapseNearNotFlippable", &MR::DecimateSettings::collapseNearNotFlippable )
        .property( "touchNearBdEdges", &MR::DecimateSettings::touchNearBdEdges )
        .property( "touchBdVerts", &MR::DecimateSettings::touchBdVerts )
        .property( "maxAngleChange", &MR::DecimateSettings::maxAngleChange )
        .property( "packMesh", &MR::DecimateSettings::packMesh )
        .property( "subdivideParts", &MR::DecimateSettings::subdivideParts )
        .property( "decimateBetweenParts", &MR::DecimateSettings::decimateBetweenParts )
        .property( "minFacesInPart", &MR::DecimateSettings::minFacesInPart );

    function( "decimateMesh", &decimateMeshWrap );
}
