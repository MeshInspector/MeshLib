#include "MRWasmBindings.h"

#include "MRMesh/MRMeshCollidePrecise.h"
#include "MRMesh/MRPrecisePredicates3.h"
#include "MRMesh/MRMesh.h"

#include <emscripten/bind.h>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_mesh_collide_precise )
{
    emscripten::class_<VarEdgeTri>( "VarEdgeTri" )
        .property( "edge", +[]( const VarEdgeTri& v ) { return (int)v.edge; } )
        .function( "tri", +[]( const VarEdgeTri& v ) { return (int)v.tri(); } )
        .function( "isEdgeATriB", +[]( const VarEdgeTri& v ) { return v.isEdgeATriB(); } );

    emscripten::class_<PreciseCollisionResult>( "PreciseCollisionResult" )
        .function( "size", +[]( const PreciseCollisionResult& r ) { return (int)r.size(); } )
        .function( "get", +[]( const PreciseCollisionResult& r, int i ) { return r[i]; } );

    emscripten::function( "getVectorConverters", +[]( const Mesh& a, const Mesh& b )
    {
        return getVectorConverters( a, b );
    } );

    emscripten::function( "findCollidingEdgeTrisPrecise", +[]( const Mesh& a, const Mesh& b, const CoordinateConverters& conv )
    {
        return findCollidingEdgeTrisPrecise( a, b, conv.toInt );
    } );
}
