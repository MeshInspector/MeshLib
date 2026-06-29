#include "MRWasmBindings.h"

#include "MRMesh/MRMeshToPointCloud.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRPointCloud.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_mesh_to_point_cloud )
{
    emscripten::function( "meshToPointCloud", +[]( std::shared_ptr<Mesh> m, bool saveNormals )
    {
        return meshToPointCloud( *m, saveNormals );
    } );
}
