#include "MRWasmBindings.h"

#include "MRMesh/MRMeshCollidePrecise.h"
#include "MRMesh/MRPrecisePredicates3.h"
#include "MRMesh/MRMesh.h"

#include <emscripten/bind.h>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_mesh_collide_precise )
{
    // Opaque carrier: std::vector<VarEdgeTri>; produced here, consumed by orderIntersectionContours.
    emscripten::class_<PreciseCollisionResult>( "PreciseCollisionResult" );

    emscripten::function( "getVectorConverters", +[]( const Mesh& a, const Mesh& b )
    {
        return getVectorConverters( a, b );
    } );

    emscripten::function( "findCollidingEdgeTrisPrecise", +[]( const Mesh& a, const Mesh& b, const CoordinateConverters& conv )
    {
        return findCollidingEdgeTrisPrecise( a, b, conv.toInt );
    } );
}
