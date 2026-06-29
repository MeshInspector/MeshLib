#include "MRWasmBindings.h"

#include "MRMesh/MRPointsToMeshProjector.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_points_to_mesh_projector )
{
    emscripten::class_<MeshProjectionParameters>( "MeshProjectionParameters" )
        .constructor<>()
        .property( "loDistLimitSq", &MeshProjectionParameters::loDistLimitSq )
        .property( "upDistLimitSq", &MeshProjectionParameters::upDistLimitSq );

    emscripten::function( "findSignedDistances",
        +[]( std::shared_ptr<Mesh> refMesh, std::shared_ptr<Mesh> mesh, const MeshProjectionParameters& params )
    {
        return Wasm::packedToTypedArray<VertScalars, float>( findSignedDistances( *refMesh, *mesh, params ) );
    } );
}
