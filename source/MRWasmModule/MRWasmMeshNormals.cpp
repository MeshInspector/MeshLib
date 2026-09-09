#include "MRWasmBindings.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshNormals.h"
#include "MRMesh/MRVector.h"
#include "MRMesh/MRVector3.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_mesh_normals )
{
    emscripten::function( "computePerVertNormals",
        +[]( std::shared_ptr<Mesh> mesh ) { return computePerVertNormals( *mesh ); } );

    emscripten::function( "computePerFaceNormals",
        +[]( std::shared_ptr<Mesh> mesh ) { return computePerFaceNormals( *mesh ); } );
}
