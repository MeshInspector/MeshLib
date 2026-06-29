#include "MRWasmBindings.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshTopology.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_mesh )
{
    emscripten::class_<Mesh>( "Mesh" )
        .smart_ptr<std::shared_ptr<Mesh>>( "MeshPtr" )
        .property( "points", +[]( const Mesh& m ) { return m.points; } )
        .property( "topology", +[]( const Mesh& m ) { return m.topology; } )
        .class_function( "fromTriangles", +[]( const VertCoords& coords, const Triangulation& t )
        {
            return std::make_shared<Mesh>( Mesh::fromTriangles( coords, t ) );
        } )
        .function( "pack", +[]( Mesh& m ) { m.pack(); } );
}
