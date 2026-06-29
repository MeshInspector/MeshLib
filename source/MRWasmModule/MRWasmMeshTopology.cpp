#include "MRWasmBindings.h"

#include "MRMesh/MRMeshTopology.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector.h"
#include "MRMesh/MRId.h"

#include <emscripten/bind.h>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_mesh_topology )
{
    emscripten::class_<MeshTopology>( "MeshTopology" )
        .function( "getTriangulation", &MeshTopology::getTriangulation );
}
