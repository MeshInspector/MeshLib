#include "MRWasmBindings.h"

#include "MRVoxels/MRPointsToMeshFusion.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRMesh.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_points_to_mesh_fusion )
{
    emscripten::class_<PointsToMeshParameters>( "PointsToMeshParameters" )
        .constructor<>()
        .property( "sigma", &PointsToMeshParameters::sigma )
        .property( "minWeight", &PointsToMeshParameters::minWeight )
        .property( "invSigmaModifier", &PointsToMeshParameters::invSigmaModifier )
        .property( "sqrtAngleWeight", &PointsToMeshParameters::sqrtAngleWeight )
        .property( "voxelSize", &PointsToMeshParameters::voxelSize );

    emscripten::function( "pointsToMeshFusion", +[]( const PointCloud& cloud, const PointsToMeshParameters& params )
    {
        return std::make_shared<Mesh>( Wasm::unwrap( pointsToMeshFusion( cloud, params ) ) );
    } );
}
