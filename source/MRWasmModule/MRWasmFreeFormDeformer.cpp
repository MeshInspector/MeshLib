#include "MRWasmBindings.h"

#include "MRMesh/MRFreeFormDeformer.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRBox.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_free_form_deformer )
{
    emscripten::class_<FreeFormDeformer>( "FreeFormDeformer" )
        .constructor( +[]( std::shared_ptr<Mesh> mesh ) { return new FreeFormDeformer( *mesh ); }, emscripten::allow_raw_pointers() )
        .function( "init", +[]( FreeFormDeformer& self ) { self.init(); } )
        .function( "init", +[]( FreeFormDeformer& self, const Vector3i& resolution ) { self.init( resolution ); } )
        .function( "init", +[]( FreeFormDeformer& self, const Vector3i& resolution, const Box3f& initialBox ) { self.init( resolution, initialBox ); } )
        .function( "setRefGridPointPosition", &FreeFormDeformer::setRefGridPointPosition )
        .function( "getRefGridPointPosition", &FreeFormDeformer::getRefGridPointPosition )
        .function( "apply", &FreeFormDeformer::apply );
}
