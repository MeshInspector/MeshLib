#include "MRWasmBindings.h"

#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRId.h"
#include "MRMesh/MRColor.h"

#include <emscripten/bind.h>

#include <cstdint>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_vector )
{
    emscripten::class_<VertCoords>( "VertCoords" )
        .class_function( "fromArray", &Wasm::packedFromTypedArray<VertCoords, float, 3> )
        .function( "toArray", &Wasm::packedToTypedArray<VertCoords, float, 3> )
        .function( "size", +[]( const VertCoords& v ) { return (int)v.size(); } )
        .function( "get", +[]( const VertCoords& v, int i ) { return v[VertId( i )]; } );

    emscripten::class_<Triangulation>( "Triangulation" )
        .class_function( "fromArray", &Wasm::packedFromTypedArray<Triangulation, uint32_t, 3> )
        .function( "toArray", &Wasm::packedToTypedArray<Triangulation, uint32_t, 3> );

    emscripten::class_<FaceNormals>( "FaceNormals" )
        .class_function( "fromArray", &Wasm::packedFromTypedArray<FaceNormals, float, 3> )
        .function( "toArray", &Wasm::packedToTypedArray<FaceNormals, float, 3> );

    emscripten::class_<VertMap>( "VertMap" )
        .class_function( "fromArray", &Wasm::packedFromTypedArray<VertMap, uint32_t> )
        .function( "toArray", &Wasm::packedToTypedArray<VertMap, uint32_t> );

    emscripten::class_<FaceMap>( "FaceMap" )
        .class_function( "fromArray", &Wasm::packedFromTypedArray<FaceMap, uint32_t> )
        .function( "toArray", &Wasm::packedToTypedArray<FaceMap, uint32_t> );

    emscripten::class_<Face2RegionMap>( "Face2RegionMap" )
        .class_function( "fromArray", &Wasm::packedFromTypedArray<Face2RegionMap, uint32_t> )
        .function( "toArray", &Wasm::packedToTypedArray<Face2RegionMap, uint32_t> );

    emscripten::class_<VertColors>( "VertColors" )
        .class_function( "fromArray", &Wasm::packedFromTypedArray<VertColors, uint8_t, 4> )
        .function( "toArray", &Wasm::packedToTypedArray<VertColors, uint8_t, 4> );
}
