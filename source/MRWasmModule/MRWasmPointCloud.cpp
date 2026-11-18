#include "MRWasmBindings.h"

#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRVector.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRId.h"

#include <emscripten/bind.h>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_point_cloud )
{
    emscripten::class_<PointCloud>( "PointCloud" )
        .constructor<>()
        .property( "points", +[]( const PointCloud& pc ) { return pc.points; } )
        .property( "normals", +[]( const PointCloud& pc ) { return pc.normals; } )
        .property( "validPoints",
            +[]( const PointCloud& pc ) { return pc.validPoints; },
            +[]( PointCloud& pc, const VertBitSet& bs ) { pc.validPoints = bs; } )
        .function( "addPoint", +[]( PointCloud& pc, const Vector3f& point ) { return (int)pc.addPoint( point ); } )
        .function( "computeBoundingBox", +[]( const PointCloud& pc ) { return pc.computeBoundingBox(); } )
        .function( "invalidateCaches", +[]( PointCloud& pc ) { pc.invalidateCaches(); } );
}
