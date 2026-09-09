#include "MRWasmBindings.h"

#include "MRMesh/MRIntersectionContour.h"
#include "MRMesh/MRMeshCollidePrecise.h"
#include "MRMesh/MRMeshTopology.h"

#include <emscripten/bind.h>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_intersection_contour )
{
    // Opaque carrier: std::vector<std::vector<VarEdgeTri>>; produced here, consumed by getOneMeshIntersectionContours.
    emscripten::class_<ContinuousContours>( "ContinuousContours" );

    emscripten::function( "orderIntersectionContours",
        +[]( const MeshTopology& topologyA, const MeshTopology& topologyB, const PreciseCollisionResult& intersections )
    {
        return orderIntersectionContours( topologyA, topologyB, intersections );
    } );
}
