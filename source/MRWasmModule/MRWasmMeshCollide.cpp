#include "MRWasmBindings.h"

#include "MRMesh/MRMeshCollide.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshPart.h"
#include "MRMesh/MRFaceFace.h"
#include "MRMesh/MRBitSet.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <memory>
#include <vector>

using namespace MR;

namespace
{

emscripten::val faceFacesToArray( const std::vector<FaceFace>& pairs )
{
    emscripten::val arr = emscripten::val::array();
    for ( const FaceFace& ff : pairs )
    {
        emscripten::val o = emscripten::val::object();
        o.set( "aFace", (int)ff.aFace );
        o.set( "bFace", (int)ff.bFace );
        arr.call<void>( "push", o );
    }
    return arr;
}

}

EMSCRIPTEN_BINDINGS( meshlib_mesh_collide )
{
    emscripten::function( "isInside", +[]( std::shared_ptr<Mesh> a, std::shared_ptr<Mesh> b )
    {
        return isInside( *a, *b );
    } );

    emscripten::function( "findCollidingTriangles", +[]( std::shared_ptr<Mesh> a, std::shared_ptr<Mesh> b )
    {
        return faceFacesToArray( findCollidingTriangles( *a, *b ) );
    } );

    emscripten::function( "findCollidingTriangles", +[]( std::shared_ptr<Mesh> a, std::shared_ptr<Mesh> b, bool firstIntersectionOnly )
    {
        return faceFacesToArray( findCollidingTriangles( *a, *b, nullptr, firstIntersectionOnly ) );
    } );

    emscripten::function( "findCollidingTriangleBitsets", +[]( std::shared_ptr<Mesh> a, std::shared_ptr<Mesh> b )
    {
        auto bitsets = findCollidingTriangleBitsets( *a, *b );
        emscripten::val out = emscripten::val::object();
        out.set( "a", bitsets.first );
        out.set( "b", bitsets.second );
        return out;
    } );

    emscripten::function( "findSelfCollidingTriangles", +[]( std::shared_ptr<Mesh> mp )
    {
        return faceFacesToArray( Wasm::unwrap( findSelfCollidingTriangles( *mp ) ) );
    } );

    emscripten::function( "findSelfCollidingTrianglesBS", +[]( std::shared_ptr<Mesh> mp )
    {
        return Wasm::unwrap( findSelfCollidingTrianglesBS( *mp ) );
    } );
}
