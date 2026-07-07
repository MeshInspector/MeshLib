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

    emscripten::function( "getCircumscribedMetric", +[]( std::shared_ptr<Mesh> m ) { return getCircumscribedMetric( *m ); } );
    emscripten::function( "getPlaneFillMetric", +[]( std::shared_ptr<Mesh> m, int e ) { return getPlaneFillMetric( *m, EdgeId( e ) ); } );
    emscripten::function( "getPlaneNormalizedFillMetric", +[]( std::shared_ptr<Mesh> m, int e ) { return getPlaneNormalizedFillMetric( *m, EdgeId( e ) ); } );
    emscripten::function( "getComplexFillMetric", +[]( std::shared_ptr<Mesh> m, int e ) { return getComplexFillMetric( *m, EdgeId( e ) ); } );
    emscripten::function( "getEdgeLengthFillMetric", +[]( std::shared_ptr<Mesh> m ) { return getEdgeLengthFillMetric( *m ); } );
    emscripten::function( "getUniversalMetric", +[]( std::shared_ptr<Mesh> m ) { return getUniversalMetric( *m ); } );
    emscripten::function( "getMinAreaMetric", +[]( std::shared_ptr<Mesh> m ) { return getMinAreaMetric( *m ); } );

    emscripten::function( "calcCombinedFillMetric", +[]( std::shared_ptr<Mesh> m, const FaceBitSet& region, const FillHoleMetric& metric )
    {
        return calcCombinedFillMetric( *m, region, metric );
    } );
}
