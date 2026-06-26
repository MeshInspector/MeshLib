#include "MRWasmBindings.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshTopology.h"
#include "MRMesh/MRMeshNormals.h"
#include "MRMesh/MRId.h"
#include "MRMesh/MRVector.h"
#include "MRMesh/MRVector3.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

using namespace MR;

namespace Wasm
{

struct Helpers {};

std::shared_ptr<Mesh> meshFromGeometry( emscripten::val positions, emscripten::val indices )
{
    const auto pos = emscripten::convertJSArrayToNumberVector<float>( positions );
    const auto idx = emscripten::convertJSArrayToNumberVector<uint32_t>( indices );

    const size_t numVerts = pos.size() / 3;
    const size_t numTris = idx.size() / 3;

    VertCoords coords;
    coords.vec_.resize( numVerts );
    if ( numVerts != 0 )
        std::memcpy( coords.vec_.data(), pos.data(), numVerts * 3 * sizeof( float ) );

    Triangulation tris;
    tris.vec_.resize( numTris );
    for ( size_t f = 0; f < numTris; ++f )
        tris.vec_[f] = ThreeVertIds{
            VertId( (int)idx[3 * f + 0] ),
            VertId( (int)idx[3 * f + 1] ),
            VertId( (int)idx[3 * f + 2] ) };

    return std::make_shared<Mesh>( Mesh::fromTriangles( std::move( coords ), tris ) );
}

emscripten::val meshToGeometry( std::shared_ptr<Mesh> meshIn, bool wantNormals )
{
    auto mesh = *meshIn;
    mesh.pack();

    const size_t numVerts = mesh.points.size();
    auto positions = toFloat32Array(
        reinterpret_cast<const float*>( mesh.points.vec_.data() ), numVerts * 3 );

    const auto tris = mesh.topology.getAllTriVerts();
    std::vector<uint32_t> flatIdx( tris.size() * 3 );
    for ( size_t f = 0; f < tris.size(); ++f )
    {
        flatIdx[3 * f + 0] = (uint32_t)(int)tris[f][0];
        flatIdx[3 * f + 1] = (uint32_t)(int)tris[f][1];
        flatIdx[3 * f + 2] = (uint32_t)(int)tris[f][2];
    }
    auto indices = toUint32Array( flatIdx.data(), flatIdx.size() );

    auto result = emscripten::val::object();
    result.set( "positions", positions );
    result.set( "indices", indices );

    if ( wantNormals )
    {
        const auto normals = computePerVertNormals( mesh );
        result.set( "normals", toFloat32Array(
            reinterpret_cast<const float*>( normals.vec_.data() ), normals.size() * 3 ) );
    }
    return result;
}

}

EMSCRIPTEN_BINDINGS( meshlib_mesh )
{
    emscripten::class_<Mesh>( "Mesh" )
        .smart_ptr<std::shared_ptr<Mesh>>( "MeshPtr" );

    emscripten::class_<Wasm::Helpers>( "Wasm" )
        .class_function( "meshFromGeometry", &Wasm::meshFromGeometry )
        .class_function( "meshToGeometry", &Wasm::meshToGeometry );
}
