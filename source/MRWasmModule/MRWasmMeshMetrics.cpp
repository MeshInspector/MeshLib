#include "MRWasmBindings.h"

#include "MRMesh/MRMeshMetrics.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRId.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_mesh_metrics )
{
    emscripten::class_<FillHoleMetric>( "FillHoleMetric" );

    emscripten::function( "getCircumscribedMetric", +[]( std::shared_ptr<Mesh> mesh ) { return getCircumscribedMetric( *mesh ); } );
    emscripten::function( "getPlaneFillMetric", +[]( std::shared_ptr<Mesh> mesh, int e ) { return getPlaneFillMetric( *mesh, EdgeId( e ) ); } );
    emscripten::function( "getPlaneNormalizedFillMetric", +[]( std::shared_ptr<Mesh> mesh, int e ) { return getPlaneNormalizedFillMetric( *mesh, EdgeId( e ) ); } );
    emscripten::function( "getComplexFillMetric", +[]( std::shared_ptr<Mesh> mesh, int e ) { return getComplexFillMetric( *mesh, EdgeId( e ) ); } );
    emscripten::function( "getEdgeLengthFillMetric", +[]( std::shared_ptr<Mesh> mesh ) { return getEdgeLengthFillMetric( *mesh ); } );
    emscripten::function( "getUniversalMetric", +[]( std::shared_ptr<Mesh> mesh ) { return getUniversalMetric( *mesh ); } );
    emscripten::function( "getMinAreaMetric", +[]( std::shared_ptr<Mesh> mesh ) { return getMinAreaMetric( *mesh ); } );

    emscripten::function( "calcCombinedFillMetric", +[]( std::shared_ptr<Mesh> mesh, const FaceBitSet& filledRegion, const FillHoleMetric& metric )
    {
        return calcCombinedFillMetric( *mesh, filledRegion, metric );
    } );
}
