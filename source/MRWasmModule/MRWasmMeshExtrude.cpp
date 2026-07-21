#include "MRWasmBindings.h"

#include "MRMesh/MRMeshExtrude.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRBitSet.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_mesh_extrude )
{
    emscripten::function( "makeDegenerateBandAroundRegion", +[]( std::shared_ptr<Mesh> mesh, const FaceBitSet& region )
    {
        makeDegenerateBandAroundRegion( *mesh, region );
    } );
}
