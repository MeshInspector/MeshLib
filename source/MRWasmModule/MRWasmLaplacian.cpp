#include "MRWasmBindings.h"

#include "MRMesh/MRLaplacian.h"
#include "MRMesh/MREnums.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRVector3.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_laplacian )
{
    emscripten::class_<Laplacian>( "Laplacian" )
        .constructor( +[]( std::shared_ptr<Mesh> mesh ) { return new Laplacian( *mesh ); }, emscripten::allow_raw_pointers() )
        .function( "init", +[]( Laplacian& self, const VertBitSet& freeVerts, EdgeWeights weights )
        {
            self.init( freeVerts, weights );
        } )
        .function( "init", +[]( Laplacian& self, const VertBitSet& freeVerts, EdgeWeights weights, VertexMass vmass, RememberShape rem )
        {
            self.init( freeVerts, weights, vmass, rem );
        } )
        .function( "fixVertex", +[]( Laplacian& self, int v, bool smooth )
        {
            self.fixVertex( VertId( v ), smooth );
        } )
        .function( "fixVertex", +[]( Laplacian& self, int v, const Vector3f& fixedPos, bool smooth )
        {
            self.fixVertex( VertId( v ), fixedPos, smooth );
        } )
        .function( "apply", &Laplacian::apply );
}
