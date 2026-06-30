#include "MRWasmBindings.h"

#include "MRMesh/MRMeshTopology.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector.h"
#include "MRMesh/MRId.h"
#include "MRMesh/MRBitSet.h"

#include <emscripten/bind.h>

#include <cstdint>
#include <vector>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_mesh_topology )
{
    emscripten::class_<MeshTopology>( "MeshTopology" )
        .function( "getTriangulation", &MeshTopology::getTriangulation )
        .function( "getValidVerts", +[]( const MeshTopology& t ) { return t.getValidVerts(); } )
        .function( "getValidFaces", +[]( const MeshTopology& t ) { return t.getValidFaces(); } )
        .function( "faceSize", +[]( const MeshTopology& t ) { return t.faceSize(); } )
        .function( "findNumHoles", +[]( const MeshTopology& t ) { return t.findNumHoles(); } )
        .function( "getTriVerts", +[]( const MeshTopology& t, int f )
        {
            return Wasm::packedToTypedArray<ThreeVertIds, uint32_t>( t.getTriVerts( FaceId( f ) ) );
        } )
        .function( "getLeftTriVerts", +[]( const MeshTopology& t, int e )
        {
            return Wasm::packedToTypedArray<ThreeVertIds, uint32_t>( t.getLeftTriVerts( EdgeId( e ) ) );
        } )
        .function( "findHoleRepresentiveEdges", +[]( const MeshTopology& t )
        {
            return Wasm::packedToTypedArray<std::vector<EdgeId>, uint32_t>( t.findHoleRepresentiveEdges() );
        } );
}
