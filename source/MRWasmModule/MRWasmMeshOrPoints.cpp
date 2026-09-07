#include "MRWasmBindings.h"

#include "MRMesh/MRMeshOrPoints.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRAffineXf3.h"

#include <emscripten/bind.h>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_mesh_or_points )
{
    emscripten::class_<MeshOrPoints>( "MeshOrPoints" )
        .class_function( "fromMesh", +[]( const Mesh& m ) { return MeshOrPoints{ m }; } )
        .class_function( "fromPoints", +[]( const PointCloud& pc ) { return MeshOrPoints{ pc }; } );

    emscripten::class_<MeshOrPointsXf>( "MeshOrPointsXf" )
        .constructor( +[]( const MeshOrPoints& obj, const AffineXf3f& xf ) { return MeshOrPointsXf{ obj, xf }; } )
        .property( "xf", &MeshOrPointsXf::xf );
}
